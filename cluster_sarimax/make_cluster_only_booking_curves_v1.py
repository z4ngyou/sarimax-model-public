from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results" / "cluster_only_lt_sarimax_parallel_v1"
OUTPUT_DIR = BASE_DIR / "results" / "cluster_only_booking_curves_v1"

PREDICTIONS_PATH = RESULT_DIR / "cluster_only_2025dec_100trial_cluster_lt_sarimax_predictions_v1.csv"
SOURCE_PATH = BASE_DIR / "input_booking_data.csv"

LONG_OUTPUT = OUTPUT_DIR / "cluster_only_booking_curves_long_v1.csv"
DATE_METRICS_OUTPUT = OUTPUT_DIR / "cluster_only_booking_curve_metrics_by_date_v1.csv"
CLUSTER_METRICS_OUTPUT = OUTPUT_DIR / "cluster_only_booking_curve_metrics_by_cluster_v1.csv"


def normalized_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def safe_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(actual_arr) | np.isnan(predicted_arr))
    actual_arr = actual_arr[mask]
    predicted_arr = predicted_arr[mask]
    if len(actual_arr) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "R2": np.nan}

    rmse = float(np.sqrt(mean_squared_error(actual_arr, predicted_arr)))
    mae = float(mean_absolute_error(actual_arr, predicted_arr))
    valid_mape = actual_arr != 0
    mape = np.nan
    if np.any(valid_mape):
        mape = float(np.mean(np.abs((actual_arr[valid_mape] - predicted_arr[valid_mape]) / actual_arr[valid_mape])) * 100)
    r2 = float(r2_score(actual_arr, predicted_arr)) if len(actual_arr) >= 2 else np.nan
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def build_curves(predictions: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["date_key"] = normalized_date(predictions["date"])
    predictions["cluster"] = predictions["cluster"].astype(int)
    predictions["leadtime"] = predictions["leadtime"].astype(int)

    source = source.copy()
    source["date_key"] = normalized_date(source["チェックイン日"])
    source["leadtime"] = source["LT"].astype(int)
    source["cluster"] = source["クラスタ"].astype(int)

    source_cumsum = source[["date_key", "leadtime", "累積利用客室合計数"]].rename(
        columns={"累積利用客室合計数": "source_actual_cumsum"}
    )

    anchor = (
        source[source["leadtime"] == 91][["date_key", "累積利用客室合計数"]]
        .drop_duplicates("date_key")
        .rename(columns={"累積利用客室合計数": "anchor_actual_cumsum_LT91"})
    )

    rows = []
    for (model_scope, cluster, date_key), group in predictions.groupby(["model_scope", "cluster", "date_key"]):
        group = group.sort_values("leadtime", ascending=False).copy()
        anchor_row = anchor[anchor["date_key"] == date_key]
        anchor_value = float(anchor_row["anchor_actual_cumsum_LT91"].iloc[0]) if not anchor_row.empty else 0.0

        actual_cumsum = anchor_value
        predicted_cumsum = anchor_value
        for _, row in group.iterrows():
            actual_cumsum += float(row["actual"])
            predicted_cumsum += float(row["predicted"])
            rows.append(
                {
                    "model_scope": model_scope,
                    "cluster": int(cluster),
                    "date": date_key,
                    "leadtime": int(row["leadtime"]),
                    "anchor_actual_cumsum_LT91": anchor_value,
                    "actual_increment": float(row["actual"]),
                    "predicted_increment": float(row["predicted"]),
                    "actual_cumsum": actual_cumsum,
                    "predicted_cumsum": predicted_cumsum,
                }
            )

    curves = pd.DataFrame(rows)
    curves = curves.merge(source_cumsum, on=["date_key", "leadtime"], how="left") if "date_key" in curves.columns else curves
    if "date_key" in curves.columns:
        curves = curves.drop(columns=["date_key"])
    return curves.sort_values(["cluster", "date", "leadtime"], ascending=[True, True, False])


def build_metrics(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_rows = []
    for (cluster, date), group in curves.groupby(["cluster", "date"]):
        metrics = safe_metrics(group["actual_cumsum"], group["predicted_cumsum"])
        date_rows.append({"cluster": cluster, "date": date, **metrics})

    cluster_rows = []
    for cluster, group in curves.groupby("cluster"):
        metrics = safe_metrics(group["actual_cumsum"], group["predicted_cumsum"])
        cluster_rows.append({"cluster": cluster, **metrics})

    return pd.DataFrame(date_rows), pd.DataFrame(cluster_rows)


def plot_curves(curves: pd.DataFrame) -> None:
    for cluster, cluster_df in curves.groupby("cluster"):
        plt.figure(figsize=(12, 7))
        for date, group in cluster_df.groupby("date"):
            group = group.sort_values("leadtime", ascending=False)
            plt.plot(group["leadtime"], group["actual_cumsum"], color="black", alpha=0.18, linewidth=1.0)
            plt.plot(group["leadtime"], group["predicted_cumsum"], alpha=0.55, linewidth=1.1, label=date)
        plt.title(f"Cluster {cluster} booking curves, LT90 to LT0")
        plt.xlabel("Lead time")
        plt.ylabel("Cumulative booked rooms")
        plt.gca().invert_xaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"cluster{cluster}_booking_curves_all_dates_v1.png", dpi=160)
        plt.close()

        for date, group in cluster_df.groupby("date"):
            group = group.sort_values("leadtime", ascending=False)
            plt.figure(figsize=(10, 5))
            plt.plot(group["leadtime"], group["actual_cumsum"], marker="o", markersize=2.5, label="actual")
            plt.plot(group["leadtime"], group["predicted_cumsum"], marker="o", markersize=2.5, label="predicted")
            plt.title(f"Cluster {cluster} booking curve {date}")
            plt.xlabel("Lead time")
            plt.ylabel("Cumulative booked rooms")
            plt.gca().invert_xaxis()
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"cluster{cluster}_booking_curve_{date.replace('-', '')}_v1.png", dpi=160)
            plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    source = pd.read_csv(SOURCE_PATH, encoding="utf-8-sig")

    curves = build_curves(predictions, source)
    date_metrics, cluster_metrics = build_metrics(curves)

    curves.to_csv(LONG_OUTPUT, index=False, encoding="utf-8-sig")
    date_metrics.to_csv(DATE_METRICS_OUTPUT, index=False, encoding="utf-8-sig")
    cluster_metrics.to_csv(CLUSTER_METRICS_OUTPUT, index=False, encoding="utf-8-sig")
    plot_curves(curves)

    print(f"予測入力: {PREDICTIONS_PATH}")
    print(f"出力フォルダ: {OUTPUT_DIR}")
    print(f"ブッキングカーブ行数: {len(curves):,}")
    print(f"対象クラスタ: {sorted(curves['cluster'].unique().tolist())}")
    print(f"対象日数: {curves.groupby('cluster')['date'].nunique().to_dict()}")
    print(f"CSV: {LONG_OUTPUT}")
    print(f"日別評価: {DATE_METRICS_OUTPUT}")
    print(f"クラスタ別評価: {CLUSTER_METRICS_OUTPUT}")
    print(cluster_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
