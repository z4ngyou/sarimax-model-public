from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sarimax_model_cluster_specific_lt_parallel_v3 import (
    DATE_EXOG_COLUMNS,
    DEFAULT_MAX_AUTO_SEASONAL_CANDIDATES,
    DEFAULT_MAX_AUTO_SEASONAL_PERIOD,
    add_date_exog_features,
    model_main_cluster_parallel,
)


BASE_DIR = Path(__file__).resolve().parent
TARGET_COLUMN = "t"
NORMAL_CLUSTERS = [0, 1, 3]


@dataclass
class ClusterSpecificConfig:
    raw_data_path: str = "展開データ_ほぼ処理完了_lt_expanded_v2_with_events_jp_clean_with_cluster.csv"
    train_start_date: str = "2025-01-01"
    train_end_date: str = "2025-11-30"
    test_start_date: str = "2025-12-01"
    test_end_date: str = "2025-12-31"
    output_dir: str = "results/cluster_specific_lt_sarimax_parallel_v3"
    n_optuna_trials: int = 100
    n_jobs: int = 8
    maxiter: int = 50
    min_train_rows: int = 20
    min_test_rows: int = 1
    save_models: bool = True
    save_plots: bool = False
    include_all_normal: bool = True
    leadtimes: Optional[list[int]] = None
    run_label: Optional[str] = "cluster_specific_2025dec_100trial_auto_s"
    tune_seasonal: bool = True
    seasonal_period_candidates: Optional[list[int]] = None
    max_auto_seasonal_period: int = DEFAULT_MAX_AUTO_SEASONAL_PERIOD
    max_auto_seasonal_candidates: int = DEFAULT_MAX_AUTO_SEASONAL_CANDIDATES


def to_absolute(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return BASE_DIR / candidate


def with_label(file_name: str, run_label: Optional[str] = None) -> str:
    return f"{run_label}_{file_name}" if run_label else file_name


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp932", "shift_jis", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def prepare_base_dataframe(path: Path) -> pd.DataFrame:
    df = read_csv_with_fallback(path)
    required = {"LTまで拡張した日付", "チェックイン日", "LT", "クラスタ", TARGET_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"必要な列がありません: {sorted(missing)}")

    df = df.copy()
    df["date_part"] = pd.to_datetime(df["チェックイン日"], errors="coerce")
    if df["date_part"].isna().any():
        parsed = df["LTまで拡張した日付"].astype(str).str.split(" ").str[0]
        df["date_part"] = df["date_part"].fillna(pd.to_datetime(parsed, errors="coerce"))
    df["leadtime"] = df["LT"].astype(int)
    df["クラスタ"] = df["クラスタ"].astype(int)
    df = add_date_exog_features(df, "date_part")
    return df


def split_by_date(df: pd.DataFrame, config: ClusterSpecificConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_start = pd.to_datetime(config.train_start_date)
    train_end = pd.to_datetime(config.train_end_date)
    test_start = pd.to_datetime(config.test_start_date)
    test_end = pd.to_datetime(config.test_end_date)
    train = df[(df["date_part"] >= train_start) & (df["date_part"] <= train_end)].copy()
    test = df[(df["date_part"] >= test_start) & (df["date_part"] <= test_end)].copy()
    return train, test


def summarize_cluster_counts(df: pd.DataFrame, config: ClusterSpecificConfig) -> pd.DataFrame:
    rows = []
    for period_name, period_df in [
        ("all", df),
        ("train", split_by_date(df, config)[0]),
        ("test", split_by_date(df, config)[1]),
    ]:
        day_counts = (
            period_df[["date_part", "クラスタ"]]
            .drop_duplicates()
            .groupby("クラスタ")
            .size()
            .rename("days")
        )
        row_counts = period_df.groupby("クラスタ").size().rename("rows")
        summary = pd.concat([day_counts, row_counts], axis=1).reset_index()
        summary["period"] = period_name
        rows.append(summary)
    return pd.concat(rows, ignore_index=True).sort_values(["period", "クラスタ"])


def build_leadtime_data(
    df: pd.DataFrame,
    config: ClusterSpecificConfig,
    cluster_filter: Optional[int] = None,
) -> dict[int, dict[str, pd.DataFrame]]:
    data = df[df["クラスタ"].isin(NORMAL_CLUSTERS)].copy()
    if cluster_filter is not None:
        data = data[data["クラスタ"] == cluster_filter].copy()

    requested = set(config.leadtimes) if config.leadtimes is not None else set(range(0, 91))
    leadtime_data: dict[int, dict[str, pd.DataFrame]] = {}

    for lt in sorted(requested, reverse=True):
        lt_df = data[data["leadtime"] == lt].copy().sort_values("date_part")
        if lt_df.empty:
            continue
        train, test = split_by_date(lt_df, config)
        if len(train) < config.min_train_rows or len(test) < config.min_test_rows:
            continue
        leadtime_data[int(lt)] = {"train": train, "test": test}

    return leadtime_data


def safe_metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    if len(actual) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "R2": np.nan}

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    valid_mape = actual != 0
    mape = np.nan
    if np.any(valid_mape):
        mape = float(np.mean(np.abs((actual[valid_mape] - predicted[valid_mape]) / actual[valid_mape])) * 100)
    r2 = float(r2_score(actual, predicted)) if len(actual) >= 2 else np.nan
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def build_all_vs_cluster_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    all_pred = predictions[predictions["model_scope"] == "all_normal"].copy()
    all_pred = all_pred.rename(columns={"predicted": "predicted_all_normal"})
    all_pred = all_pred[["date", "cluster", "leadtime", "actual", "predicted_all_normal"]]

    cluster_pred = predictions[predictions["model_scope"].str.startswith("cluster_")].copy()
    cluster_pred = cluster_pred.rename(columns={"predicted": "predicted_cluster_model"})
    cluster_pred = cluster_pred[["date", "cluster", "leadtime", "actual", "predicted_cluster_model", "model_scope"]]

    comparison = cluster_pred.merge(
        all_pred,
        on=["date", "cluster", "leadtime", "actual"],
        how="left",
    )
    comparison["error_all_normal"] = comparison["actual"] - comparison["predicted_all_normal"]
    comparison["error_cluster_model"] = comparison["actual"] - comparison["predicted_cluster_model"]
    comparison["abs_error_all_normal"] = comparison["error_all_normal"].abs()
    comparison["abs_error_cluster_model"] = comparison["error_cluster_model"].abs()
    return comparison.sort_values(["cluster", "date", "leadtime"], ascending=[True, True, False])


def build_metrics_by_cluster(predictions: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_normal = predictions[predictions["model_scope"] == "all_normal"]
    if not all_normal.empty:
        rows.append({"cluster": "0_1_3", "model_scope": "all_normal", **safe_metrics(all_normal["actual"], all_normal["predicted"])})

    for cluster in NORMAL_CLUSTERS:
        cluster_scope = f"cluster_{cluster}"
        cluster_predictions = predictions[predictions["model_scope"] == cluster_scope]
        if not comparison.empty:
            all_rows = comparison[comparison["cluster"] == cluster]
            rows.append(
                {
                    "cluster": cluster,
                    "model_scope": "all_normal_on_same_cluster_dates",
                    **safe_metrics(all_rows["actual"], all_rows["predicted_all_normal"]),
                }
            )
            rows.append(
                {
                    "cluster": cluster,
                    "model_scope": cluster_scope,
                    **safe_metrics(all_rows["actual"], all_rows["predicted_cluster_model"]),
                }
            )
        elif not cluster_predictions.empty:
            rows.append(
                {
                    "cluster": cluster,
                    "model_scope": cluster_scope,
                    **safe_metrics(cluster_predictions["actual"], cluster_predictions["predicted"]),
                }
            )
    return pd.DataFrame(rows)


def plot_cluster_lt0_comparison(comparison: pd.DataFrame, output_dir: Path, run_label: Optional[str]) -> None:
    if comparison.empty or "leadtime" not in comparison.columns:
        return

    lt0 = comparison[comparison["leadtime"] == 0].copy()
    if lt0.empty:
        return

    for cluster in NORMAL_CLUSTERS:
        data = lt0[lt0["cluster"] == cluster].sort_values("date")
        if data.empty:
            continue
        plt.figure(figsize=(10, 5))
        plt.plot(pd.to_datetime(data["date"]), data["actual"], marker="o", label="actual")
        plt.plot(pd.to_datetime(data["date"]), data["predicted_all_normal"], marker="o", label="all_normal")
        plt.plot(pd.to_datetime(data["date"]), data["predicted_cluster_model"], marker="o", label=f"cluster_{cluster}")
        plt.title(f"Cluster {cluster} LT0 actual vs predictions")
        plt.xlabel("date")
        plt.ylabel(TARGET_COLUMN)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / with_label(f"cluster{cluster}_lt0_actual_vs_pred.png", run_label))
        plt.close()


def run_cluster_specific_pipeline(config: Optional[ClusterSpecificConfig] = None) -> None:
    config = config or ClusterSpecificConfig()
    raw_data_path = to_absolute(config.raw_data_path)
    output_dir = to_absolute(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_base_dataframe(raw_data_path)
    df_normal = df[df["クラスタ"].isin(NORMAL_CLUSTERS)].copy()
    cluster_counts = summarize_cluster_counts(df, config)
    cluster_counts.to_csv(output_dir / with_label("cluster_counts.csv", config.run_label), index=False, encoding="utf-8-sig")

    print(f"入力ファイル名: {raw_data_path}")
    print(f"出力フォルダ: {output_dir}")
    print(f"行数: {len(df):,}")
    print(f"チェックイン日の最小値: {df['date_part'].min().date()}")
    print(f"チェックイン日の最大値: {df['date_part'].max().date()}")
    print(f"LTの最小値: {df['leadtime'].min()}")
    print(f"LTの最大値: {df['leadtime'].max()}")
    print(f"クラスタ列の有無: {'クラスタ' in df.columns}")
    print(f"クラスタ2の件数: {(df['クラスタ'] == 2).sum():,}")
    print(f"除外後の件数: {len(df_normal):,}")
    print(f"学習期間: {config.train_start_date} to {config.train_end_date}")
    print(f"テスト期間: {config.test_start_date} to {config.test_end_date}")
    print(f"使用する外生変数: {DATE_EXOG_COLUMNS}")
    print(f"seasonal_orderのOptuna探索: {config.tune_seasonal}")
    if config.seasonal_period_candidates is None:
        print(
            "Optunaで探索するs候補: "
            f"各クラスタ×LTの学習系列から自動抽出 "
            f"(最大s={config.max_auto_seasonal_period}, 最大候補数={config.max_auto_seasonal_candidates})"
        )
    else:
        print(f"Optunaで探索するs候補: 明示指定 {config.seasonal_period_candidates}")
    print("既定LT範囲: LT90を含む range(0, 91)")
    print("クラスタごとの件数:")
    print(cluster_counts.to_string(index=False))

    all_results = []
    all_predictions = []

    scopes: list[tuple[str, Optional[int]]] = []
    if config.include_all_normal:
        scopes.append(("all_normal", None))
    scopes.extend((f"cluster_{cluster}", cluster) for cluster in NORMAL_CLUSTERS)

    for model_scope, cluster_filter in scopes:
        print(f"=== {model_scope} を学習します ===")
        leadtime_data = build_leadtime_data(df, config, cluster_filter=cluster_filter)
        print(f"{model_scope} 学習対象LT数: {len(leadtime_data)}")
        if config.leadtimes is None or 90 in set(config.leadtimes):
            status = "含まれています" if 90 in leadtime_data else "含まれていません（データ不足またはフィルタ条件）"
            print(f"{model_scope} LT90の学習・予測対象: {status}")
        results_df, predictions_df = model_main_cluster_parallel(
            leadtime_data,
            TARGET_COLUMN,
            model_scope=model_scope,
            output_dir=str(output_dir),
            n_trials=config.n_optuna_trials,
            n_jobs=config.n_jobs,
            maxiter=config.maxiter,
            save_models=config.save_models,
            save_plots=config.save_plots,
            leadtimes=config.leadtimes,
            run_label=config.run_label,
            tune_seasonal=config.tune_seasonal,
            seasonal_period_candidates=config.seasonal_period_candidates,
            max_auto_seasonal_period=config.max_auto_seasonal_period,
            max_auto_seasonal_candidates=config.max_auto_seasonal_candidates,
        )
        all_results.append(results_df)
        all_predictions.append(predictions_df)

    metrics = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    comparison = (
        build_all_vs_cluster_comparison(predictions)
        if config.include_all_normal and not predictions.empty
        else pd.DataFrame()
    )
    metrics_by_cluster = build_metrics_by_cluster(predictions, comparison) if not predictions.empty else pd.DataFrame()

    metrics_path = output_dir / with_label("cluster_lt_sarimax_metrics_v3.csv", config.run_label)
    predictions_path = output_dir / with_label("cluster_lt_sarimax_predictions_v3.csv", config.run_label)
    comparison_path = output_dir / with_label("cluster_lt_sarimax_comparison_all_vs_cluster_v3.csv", config.run_label)
    metrics_by_cluster_path = output_dir / with_label("cluster_lt_sarimax_metrics_by_cluster_v3.csv", config.run_label)

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    metrics_by_cluster.to_csv(metrics_by_cluster_path, index=False, encoding="utf-8-sig")
    plot_cluster_lt0_comparison(comparison, output_dir, config.run_label)

    print(f"予測結果: {predictions_path}")
    print(f"LT別評価: {metrics_path}")
    print(f"全体モデルvsクラスタモデル比較: {comparison_path}")
    print(f"クラスタ別評価: {metrics_by_cluster_path}")
    if not metrics_by_cluster.empty:
        print("クラスタ別評価指標:")
        print(metrics_by_cluster.to_string(index=False))


if __name__ == "__main__":
    run_cluster_specific_pipeline()
