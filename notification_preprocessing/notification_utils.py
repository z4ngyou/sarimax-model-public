from collections import defaultdict
from pathlib import Path
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
NOTIFICATION_DIR = BASE_DIR / "data" / "notifications"

RESERVATION_ID_COLUMN = "reservation_id"
NOTIFICATION_ID_COLUMN = "notification_id"
NOTIFICATION_TYPE_COLUMN = "notification_type"
RECEIVED_AT_COLUMN = "received_at"
STAY_DATE_PREFIX = "stay_date_"
ROOM_COUNT_PREFIX = "room_count_"
SOURCE_FILE_COLUMN = "source_file"
CANCEL_NOTIFICATION_TYPE = "cancel"


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return BASE_DIR / path


def read_csv_any_encoding(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp932", "shift-jis"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode CSV: {path}")


def load_notification_files(notification_dir: Path) -> pd.DataFrame:
    csv_files = sorted(notification_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {notification_dir}")

    frames = []
    for csv_file in csv_files:
        frame = read_csv_any_encoding(csv_file)
        frame[SOURCE_FILE_COLUMN] = csv_file.name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.loc[:, ~combined.columns.astype(str).str.startswith("Unnamed:")].copy()
    return combined


def prepare_notifications(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        RESERVATION_ID_COLUMN,
        NOTIFICATION_ID_COLUMN,
        NOTIFICATION_TYPE_COLUMN,
        RECEIVED_AT_COLUMN,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    prepared = frame.copy()
    for column in (RESERVATION_ID_COLUMN, NOTIFICATION_ID_COLUMN, NOTIFICATION_TYPE_COLUMN):
        prepared[column] = prepared[column].astype(str).str.strip()
    prepared[RECEIVED_AT_COLUMN] = pd.to_datetime(prepared[RECEIVED_AT_COLUMN], errors="coerce")
    prepared = prepared.dropna(
        subset=[RESERVATION_ID_COLUMN, NOTIFICATION_ID_COLUMN, RECEIVED_AT_COLUMN]
    ).copy()

    if SOURCE_FILE_COLUMN not in prepared.columns:
        prepared[SOURCE_FILE_COLUMN] = ""

    prepared = prepared.sort_values(
        [RESERVATION_ID_COLUMN, RECEIVED_AT_COLUMN, NOTIFICATION_ID_COLUMN, SOURCE_FILE_COLUMN]
    ).drop_duplicates(
        subset=[RESERVATION_ID_COLUMN, NOTIFICATION_ID_COLUMN],
        keep="last",
    )
    return prepared.reset_index(drop=True)


def find_stay_slots(columns: list[str]) -> list[str]:
    pattern = re.compile(rf"^{re.escape(STAY_DATE_PREFIX)}(.+)$")
    slots: list[str] = []
    for column in columns:
        match = pattern.match(str(column))
        if match:
            slots.append(match.group(1))
    return slots


def extract_state_map(row: pd.Series, stay_slots: list[str]) -> dict[pd.Timestamp, float]:
    state: dict[pd.Timestamp, float] = defaultdict(float)
    for slot in stay_slots:
        stay_date = pd.to_datetime(row.get(f"{STAY_DATE_PREFIX}{slot}"), errors="coerce")
        room_count = pd.to_numeric(row.get(f"{ROOM_COUNT_PREFIX}{slot}"), errors="coerce")
        if pd.isna(stay_date) or pd.isna(room_count) or float(room_count) == 0:
            continue
        state[pd.Timestamp(stay_date.normalize())] += float(room_count)
    return {date: rooms for date, rooms in state.items() if rooms != 0}


def build_delta_rows(frame: pd.DataFrame) -> pd.DataFrame:
    stay_slots = find_stay_slots(frame.columns.tolist())
    if not stay_slots:
        raise ValueError("No stay-date columns were found in the notification files.")

    delta_rows: list[dict] = []
    for reservation_id, group in frame.groupby(RESERVATION_ID_COLUMN, sort=False):
        previous_state: dict[pd.Timestamp, float] = {}
        ordered = group.sort_values([RECEIVED_AT_COLUMN, NOTIFICATION_ID_COLUMN])
        for _, row in ordered.iterrows():
            notification_type = str(row[NOTIFICATION_TYPE_COLUMN]).strip()
            received_at = pd.Timestamp(row[RECEIVED_AT_COLUMN]).normalize()
            current_state = extract_state_map(row, stay_slots)

            if notification_type == CANCEL_NOTIFICATION_TYPE:
                base_state = previous_state if previous_state else current_state
                delta_state = {date: -rooms for date, rooms in base_state.items()}
                previous_state = {}
            else:
                all_dates = sorted(set(previous_state) | set(current_state))
                delta_state = {}
                for stay_date in all_dates:
                    delta = current_state.get(stay_date, 0.0) - previous_state.get(stay_date, 0.0)
                    if delta != 0:
                        delta_state[stay_date] = delta
                previous_state = current_state

            for stay_date, room_delta in delta_state.items():
                delta_rows.append(
                    {
                        RESERVATION_ID_COLUMN: reservation_id,
                        NOTIFICATION_ID_COLUMN: row[NOTIFICATION_ID_COLUMN],
                        NOTIFICATION_TYPE_COLUMN: notification_type,
                        "received_date": received_at,
                        "checkin_date": stay_date,
                        "leadtime": int((stay_date - received_at).days),
                        "room_delta": room_delta,
                        SOURCE_FILE_COLUMN: row[SOURCE_FILE_COLUMN],
                    }
                )
    return pd.DataFrame(delta_rows)


def aggregate_delta_rows(delta_frame: pd.DataFrame) -> pd.DataFrame:
    if delta_frame.empty:
        return pd.DataFrame(
            columns=[
                "checkin_date",
                "leadtime",
                "room_delta",
                "event_count",
                "received_date_min",
                "received_date_max",
            ]
        )

    return (
        delta_frame.groupby(["checkin_date", "leadtime"], as_index=False)
        .agg(
            room_delta=("room_delta", "sum"),
            event_count=("room_delta", "size"),
            received_date_min=("received_date", "min"),
            received_date_max=("received_date", "max"),
        )
        .sort_values(["checkin_date", "leadtime"], ascending=[True, False])
        .reset_index(drop=True)
    )


def save_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
