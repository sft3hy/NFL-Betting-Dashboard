import requests
import pandas as pd
from typing import List, Dict, Optional, Set
from datetime import datetime, timezone, timedelta

ESPN_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
INJURIES_BASE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/injuries"

position_weights = {
    "QB": 10.0,
    "RB": 4.0,
    "WR": 3.5,
    "TE": 3.0,
    "LT": 4.5,
    "RT": 3.5,
    "LG": 3.0,
    "RG": 3.0,
    "C": 3.5,
    "DE": 4.0,
    "DT": 3.5,
    "LB": 3.5,
    "CB": 4.0,
    "S": 3.5,
    "K": 2.0,
    "P": 1.0,
}

SESSION = requests.Session()
SESSION.headers.update(
    {"User-Agent": "espn-injuries-script/1.0 (+https://example.com)"}
)


def fetch_json(url: str, timeout: int = 10) -> Optional[dict]:
    try:
        r = SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def teams_df_from_json(data: dict) -> pd.DataFrame:
    rows = []
    for sport in data.get("sports", []):
        for league in sport.get("leagues", []):
            for team_entry in league.get("teams", []):
                team = team_entry.get("team", team_entry)
                if not team or not isinstance(team, dict):
                    continue
                team_id = team.get("id")
                abbr = team.get("abbreviation")
                if team_id and abbr:
                    rows.append({"abbreviation": abbr, "id": team_id})
    return pd.DataFrame(rows, columns=["abbreviation", "id"])


def get_nfl_team_abbrev_id_df() -> pd.DataFrame:
    data = fetch_json(ESPN_TEAMS_URL)
    if data is None:
        raise RuntimeError("Failed to fetch teams from ESPN")
    df = teams_df_from_json(data)
    return df.drop_duplicates().sort_values(by="abbreviation").reset_index(drop=True)


def normalize_ref_url(ref: str) -> str:
    return ref


def extract_refs_from_injuries_page(injury_list_page: dict) -> List[str]:
    refs = []
    if not injury_list_page:
        return refs
    for item in injury_list_page.get("items", []):
        if isinstance(item, dict):
            ref = item.get("$ref") or item.get("ref")
            if ref:
                refs.append(normalize_ref_url(ref))
    return refs


def parse_iso_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        if dt_str.endswith("Z"):
            dt_str = dt_str[:-1] + "+00:00"
        return datetime.fromisoformat(dt_str)
    except ValueError:
        try:
            return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f%z")
        except ValueError:
            return None


def extract_athlete_id_from_ref(ref: str) -> Optional[str]:
    if not isinstance(ref, str):
        return None
    parts = ref.split("/")
    if "athletes" in parts:
        try:
            athlete_index = parts.index("athletes") + 1
            if athlete_index < len(parts):
                return parts[athlete_index].split("?")[0]
        except (ValueError, IndexError):
            return None
    return None


def injury_record_key(injury_obj: dict) -> Optional[str]:
    if not injury_obj:
        return None
    athlete = injury_obj.get("athlete")
    if isinstance(athlete, dict):
        if "id" in athlete:
            return str(athlete["id"])
        if "$ref" in athlete:
            return extract_athlete_id_from_ref(athlete["$ref"])
    if isinstance(athlete, str) and athlete.startswith("http"):
        return extract_athlete_id_from_ref(athlete)
    if "athleteId" in injury_obj:
        return str(injury_obj["athleteId"])
    return None


def status_indicates_active(status_text: Optional[str]) -> bool:
    if not status_text:
        return False
    s = status_text.strip().lower()
    active_indicators = [
        "active",
        "activated",
        "returned",
        "clear",
        "cleared",
        "full participant",
    ]
    return any(word in s for word in active_indicators)


def status_indicates_injured(
    status_text: Optional[str], type_obj: Optional[dict]
) -> bool:
    if status_text:
        s = status_text.strip().lower()
        injured_indicators = [
            "questionable",
            "out",
            "doubtful",
            "injured reserve",
            "day-to-day",
            "week-to-week",
        ]
        if any(word in s for word in injured_indicators):
            return True

    if isinstance(type_obj, dict):
        for key in ("name", "abbreviation"):
            v = type_obj.get(key)
            if isinstance(v, str):
                vv = v.strip().lower()
                # Exact matches for common abbreviations
                if vv in ["q", "o", "d", "ir"]:
                    return True
    return False


def get_athlete_position_from_injury(injury_obj: dict) -> Optional[str]:
    if not injury_obj:
        return None
    athlete = injury_obj.get("athlete")
    if isinstance(athlete, dict):
        pos = athlete.get("position")
        if isinstance(pos, dict):
            abbr = pos.get("abbreviation")
            if abbr:
                return str(abbr).upper()

    athlete_ref = None
    if isinstance(athlete, dict) and "$ref" in athlete:
        athlete_ref = athlete["$ref"]
    elif isinstance(athlete, str) and athlete.startswith("http"):
        athlete_ref = athlete

    if athlete_ref:
        athlete_json = fetch_json(athlete_ref)
        if athlete_json and isinstance(athlete_json.get("position"), dict):
            abbr = athlete_json["position"].get("abbreviation")
            if abbr:
                return str(abbr).upper()
    return None


def get_positions_for_team_injuries(
    team_id: str, season: Optional[int] = None, recency_days: int = 35
) -> Set[str]:
    url = INJURIES_BASE.format(team_id=team_id)
    if season:
        url = f"{url}?season={season}"

    page = fetch_json(url)
    if not page:
        return set()

    refs = extract_refs_from_injuries_page(page)
    athlete_latest: Dict[str, dict] = {}

    for ref in refs:
        injury_obj = fetch_json(ref)
        if not injury_obj:
            continue

        athlete_id = injury_record_key(injury_obj)
        if not athlete_id:
            continue

        date_str = injury_obj.get("date") or injury_obj.get("lastUpdate")
        parsed_date = parse_iso_datetime(date_str)
        if not parsed_date:
            continue

        # *** START OF KEY LOGIC CHANGE ***
        # 1. Recency Filter: Ignore old injury reports
        if parsed_date < datetime.now(timezone.utc) - timedelta(days=recency_days):
            continue

        if (
            athlete_id not in athlete_latest
            or parsed_date > athlete_latest[athlete_id]["date"]
        ):
            athlete_latest[athlete_id] = {"obj": injury_obj, "date": parsed_date}

    positions: Set[str] = set()
    for _, rec in athlete_latest.items():
        injury_obj = rec["obj"]
        status_text = injury_obj.get("status")
        type_obj = injury_obj.get("type")

        # 2. Refined Status Checks
        if status_indicates_active(status_text):
            continue
        if not status_indicates_injured(status_text, type_obj):
            continue
        # *** END OF KEY LOGIC CHANGE ***

        pos = get_athlete_position_from_injury(injury_obj)
        if pos and pos in position_weights:
            positions.add(pos)

    return positions


def build_team_injuries_dataframe(
    teams_df: pd.DataFrame, season: Optional[int] = None
) -> pd.DataFrame:
    records = []
    for _, row in teams_df.iterrows():
        team_abbr = row["abbreviation"]
        team_id = str(row["id"])
        print(f"Fetching injuries for {team_abbr}...")
        positions = get_positions_for_team_injuries(team_id=team_id, season=season)
        pos_list = sorted(list(positions))
        records.append(
            {"abbreviation": team_abbr, "id": team_id, "noteworthy_injuries": pos_list}
        )
    return pd.DataFrame(records, columns=["abbreviation", "id", "noteworthy_injuries"])


def get_nfl_teams_with_injuries_df(season: Optional[int] = None) -> pd.DataFrame:
    teams_df = get_nfl_team_abbrev_id_df()
    inj_df = build_team_injuries_dataframe(teams_df, season=season)
    return inj_df


if __name__ == "__main__":
    season_year = 2025
    df = get_nfl_teams_with_injuries_df(season=season_year)
    print("\n--- Final Results ---")
    print(df.to_string(index=False))
    # df.to_csv("csvs/nfl_teams_with_noteworthy_injuries_v2.csv", index=False)
