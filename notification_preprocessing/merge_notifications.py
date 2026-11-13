import argparse

from notification_utils import (
    NOTIFICATION_DIR,
    aggregate_delta_rows,
    build_delta_rows,
    load_notification_files,
    prepare_notifications,
    resolve_path,
    save_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge notification files and convert state changes into room deltas."
    )
    parser.add_argument(
        "--notification-dir",
        default=str(NOTIFICATION_DIR),
        help="Directory containing notification CSV files",
    )
    parser.add_argument(
        "--output-file",
        default="data/notification_deltas.csv",
        help="Output CSV path for notification-level room deltas",
    )
    parser.add_argument(
        "--aggregated-output-file",
        default="data/notification_deltas_by_checkin_leadtime.csv",
        help="Output CSV path for deltas grouped by check-in date and lead time",
    )
    args = parser.parse_args()

    notification_dir = resolve_path(args.notification_dir)
    output_file = resolve_path(args.output_file)
    aggregated_output_file = resolve_path(args.aggregated_output_file)

    combined = load_notification_files(notification_dir)
    notifications = prepare_notifications(combined)
    delta_rows = build_delta_rows(notifications)
    aggregated = aggregate_delta_rows(delta_rows)

    save_csv(delta_rows, output_file)
    save_csv(aggregated, aggregated_output_file)

    print(f"Loaded source rows: {len(combined)}")
    print(f"Distinct notifications: {len(notifications)}")
    print(f"Delta rows: {len(delta_rows)}")
    print(f"Aggregated rows: {len(aggregated)}")
    print(f"Delta CSV: {output_file}")
    print(f"Aggregated CSV: {aggregated_output_file}")


if __name__ == "__main__":
    main()
