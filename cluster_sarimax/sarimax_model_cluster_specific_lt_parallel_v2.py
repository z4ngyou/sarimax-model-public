import math
import multiprocessing as mp
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


DATE_EXOG_COLUMNS = [
    "month_sin",
    "month_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "is_mon",
    "is_tue",
    "is_wed",
    "is_thu",
    "is_fri",
    "is_sat",
]

NO_SEASONAL_ORDER = (0, 0, 0, 0)
DEFAULT_SEASONAL_PERIOD_CANDIDATES = [0, 7, 14, 30]


def add_date_exog_features(df: pd.DataFrame, date_column: str = "date_part") -> pd.DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df[date_column])
    df["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
    df["dayofyear_sin"] = np.sin(2 * np.pi * dates.dt.dayofyear / 366)
    df["dayofyear_cos"] = np.cos(2 * np.pi * dates.dt.dayofyear / 366)

    dayofweek = dates.dt.dayofweek
    weekday_map = {
        "is_mon": 0,
        "is_tue": 1,
        "is_wed": 2,
        "is_thu": 3,
        "is_fri": 4,
        "is_sat": 5,
    }
    for column, value in weekday_map.items():
        df[column] = (dayofweek == value).astype(int)
    return df


def get_exog_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in DATE_EXOG_COLUMNS if col in df.columns]


def _prepare_model_frame(data: pd.DataFrame, target_column: str):
    data_copy = data.copy().sort_values("date_part")
    exog_columns = get_exog_columns(data_copy)
    if not exog_columns:
        raise ValueError("日付由来の外生変数列が見つかりません。")

    indexed = data_copy.set_index("date_part")
    exog = data_copy[exog_columns].fillna(0).astype(float)
    exog.index = indexed.index
    y = indexed[target_column].astype(float)
    return indexed, y, exog, exog_columns


def fit_sarimax_model(
    train_data: pd.DataFrame,
    target_column: str,
    order=(1, 1, 1),
    seasonal_order=NO_SEASONAL_ORDER,
    maxiter: int = 50,
):
    _, y, exog, _ = _prepare_model_frame(train_data, target_column)
    model = SARIMAX(
        y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=maxiter)


def make_predictions(model_results, steps: int, test_data: pd.DataFrame, target_column: str):
    test_indexed, _, exog_forecast, _ = _prepare_model_frame(test_data, target_column)
    forecast = model_results.get_forecast(steps=steps, exog=exog_forecast)
    return forecast.predicted_mean, forecast.conf_int(), test_indexed


def evaluate_model(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    if len(actual) == 0:
        return {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "R2": np.nan}

    mse = mean_squared_error(actual, predicted)
    valid_mape = actual != 0
    mape = np.nan
    if np.any(valid_mape):
        mape = float(np.mean(np.abs((actual[valid_mape] - predicted[valid_mape]) / actual[valid_mape])) * 100)
    r2 = r2_score(actual, predicted) if len(actual) >= 2 else np.nan
    return {
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "MAE": mean_absolute_error(actual, predicted),
        "MAPE": mape,
        "R2": r2,
    }


def _with_label(file_name: str, run_label: str | None = None) -> str:
    return f"{run_label}_{file_name}" if run_label else file_name


def _normalize_seasonal_period_candidates(
    candidates: list[int] | None,
    n_train: int,
) -> list[int]:
    raw_candidates = candidates or DEFAULT_SEASONAL_PERIOD_CANDIDATES
    normalized = {0}
    for candidate in raw_candidates:
        s = int(candidate)
        if s == 0:
            normalized.add(0)
        elif s > 1 and n_train >= (2 * s + 5):
            normalized.add(s)
    return sorted(normalized)


def _seasonal_order_from_params(params: dict) -> tuple[int, int, int, int]:
    s = int(params.get("s", 0))
    if s <= 1:
        return NO_SEASONAL_ORDER
    seasonal_order = (
        int(params.get("P", 0)),
        int(params.get("D", 0)),
        int(params.get("Q", 0)),
        s,
    )
    if sum(seasonal_order[:3]) == 0:
        return NO_SEASONAL_ORDER
    return seasonal_order


def _complete_params(params: dict) -> dict:
    completed = {
        "p": int(params.get("p", 1)),
        "d": int(params.get("d", 1)),
        "q": int(params.get("q", 1)),
        "P": int(params.get("P", 0)),
        "D": int(params.get("D", 0)),
        "Q": int(params.get("Q", 0)),
        "s": int(params.get("s", 0)),
    }
    if _seasonal_order_from_params(completed) == NO_SEASONAL_ORDER:
        completed.update({"P": 0, "D": 0, "Q": 0, "s": 0})
    return completed


def objective(
    trial,
    train_data,
    test_data,
    target_column,
    maxiter,
    tune_seasonal: bool,
    seasonal_period_candidates: list[int] | None,
):
    p = trial.suggest_int("p", 0, 5)
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 5)

    if p + d + q > 9:
        return float("inf")

    P = D = Q = s = 0
    if tune_seasonal:
        s_choices = _normalize_seasonal_period_candidates(seasonal_period_candidates, len(train_data))
        s = trial.suggest_categorical("s", s_choices)
        if s > 0:
            P = trial.suggest_int("P", 0, 1)
            D = trial.suggest_int("D", 0, 1)
            Q = trial.suggest_int("Q", 0, 1)
            if P + D + Q == 0:
                return float("inf")
            if p + d + q + P + D + Q > 10:
                return float("inf")

    seasonal_order = _seasonal_order_from_params({"P": P, "D": D, "Q": Q, "s": s})

    try:
        model_results = fit_sarimax_model(
            train_data,
            target_column,
            order=(p, d, q),
            seasonal_order=seasonal_order,
            maxiter=maxiter,
        )
        forecast_mean, _, test_data_indexed = make_predictions(
            model_results,
            len(test_data),
            test_data,
            target_column,
        )
        mse = mean_squared_error(test_data_indexed[target_column], forecast_mean)
        return math.sqrt(mse)
    except Exception:
        return float("inf")


def optimize_sarimax_parameters(
    train_data,
    test_data,
    target_column,
    n_trials=30,
    maxiter: int = 50,
    tune_seasonal: bool = True,
    seasonal_period_candidates: list[int] | None = None,
):
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(
            trial,
            train_data,
            test_data,
            target_column,
            maxiter,
            tune_seasonal,
            seasonal_period_candidates,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    if study.best_value == float("inf"):
        return {"p": 1, "d": 1, "q": 1, "P": 0, "D": 0, "Q": 0, "s": 0}, float("inf"), None

    params = _complete_params(dict(study.best_params))
    return params, study.best_value, study.best_trial.number


def plot_results(
    test_data_indexed,
    target_column,
    forecast_mean,
    leadtime,
    output_dir,
    model_scope: str,
    run_label: str | None = None,
):
    plt.figure(figsize=(10, 5))
    plt.plot(test_data_indexed.index, test_data_indexed[target_column], label="actual")
    plt.plot(test_data_indexed.index, forecast_mean, color="red", label="predicted")
    plt.title(f"{model_scope} LT{leadtime} SARIMAX")
    plt.xlabel("date")
    plt.ylabel(target_column)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, _with_label(f"{model_scope}_LT{leadtime}_actual_vs_pred.png", run_label)))
    plt.close()


def process_leadtime_task(task):
    (
        lt,
        data,
        target_column,
        output_dir,
        n_trials,
        maxiter,
        save_models,
        save_plots,
        run_label,
        model_scope,
        tune_seasonal,
        seasonal_period_candidates,
    ) = task
    train_data = data["train"]
    test_data = data["test"]
    exog_columns = get_exog_columns(train_data)

    best_params, best_validation_rmse, best_trial_number = optimize_sarimax_parameters(
        train_data,
        test_data,
        target_column,
        n_trials=n_trials,
        maxiter=maxiter,
        tune_seasonal=tune_seasonal,
        seasonal_period_candidates=seasonal_period_candidates,
    )
    order = (best_params["p"], best_params["d"], best_params["q"])
    seasonal_order = _seasonal_order_from_params(best_params)
    model_results = fit_sarimax_model(
        train_data,
        target_column,
        order=order,
        seasonal_order=seasonal_order,
        maxiter=maxiter,
    )
    forecast_mean, conf_int, test_data_indexed = make_predictions(
        model_results,
        len(test_data),
        test_data,
        target_column,
    )
    metrics = evaluate_model(test_data_indexed[target_column], forecast_mean)

    model_path = ""
    if save_models:
        model_path = os.path.join(output_dir, _with_label(f"{model_scope}_sarimax_model_LT{lt}.pkl", run_label))
        with open(model_path, "wb") as f:
            pickle.dump(model_results, f)

    if save_plots:
        plot_results(
            test_data_indexed,
            target_column,
            forecast_mean,
            lt,
            output_dir,
            model_scope,
            run_label,
        )

    predictions_df = pd.DataFrame(
        {
            "model_scope": model_scope,
            "date": test_data_indexed.index,
            "cluster": test_data_indexed["クラスタ"].to_numpy() if "クラスタ" in test_data_indexed.columns else np.nan,
            "leadtime": lt,
            "actual": test_data_indexed[target_column].to_numpy(),
            "predicted": np.asarray(forecast_mean),
            "lower_bound": conf_int.iloc[:, 0].to_numpy(),
            "upper_bound": conf_int.iloc[:, 1].to_numpy(),
        }
    )
    predictions_df["error"] = predictions_df["actual"] - predictions_df["predicted"]
    predictions_df["abs_error"] = predictions_df["error"].abs()

    return {
        "leadtime": lt,
        "params": best_params,
        "best_validation_RMSE": best_validation_rmse,
        "best_trial_number": best_trial_number,
        "metrics": metrics,
        "predictions": predictions_df,
        "seasonal_order": seasonal_order,
        "exog_columns": exog_columns,
        "model_path": model_path,
        "n_train": len(train_data),
        "n_test": len(test_data),
    }


def model_main_cluster_parallel(
    leadtime_data,
    target_column,
    model_scope: str,
    output_dir="results/cluster_specific_lt_sarimax",
    n_trials: int = 30,
    n_jobs: int | None = None,
    maxiter: int = 50,
    save_models: bool = True,
    save_plots: bool = False,
    leadtimes: list[int] | None = None,
    run_label: str | None = None,
    tune_seasonal: bool = True,
    seasonal_period_candidates: list[int] | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    requested_leadtimes = set(leadtimes) if leadtimes else None
    filtered_leadtime_data = {
        int(lt): data
        for lt, data in leadtime_data.items()
        if requested_leadtimes is None or int(lt) in requested_leadtimes
    }
    if not filtered_leadtime_data:
        return pd.DataFrame(), pd.DataFrame()

    n_jobs = n_jobs or max(1, min(8, (os.cpu_count() or 1) - 1))
    tasks = [
        (
            lt,
            data,
            target_column,
            output_dir,
            n_trials,
            maxiter,
            save_models,
            save_plots,
            run_label,
            model_scope,
            tune_seasonal,
            seasonal_period_candidates,
        )
        for lt, data in sorted(filtered_leadtime_data.items(), reverse=True)
    ]

    result_rows = []
    prediction_frames = []
    partial_path = os.path.join(output_dir, _with_label(f"{model_scope}_partial_params.csv", run_label))

    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(process_leadtime_task, task): task[0] for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{model_scope} LT processing"):
            lt = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"{model_scope} LT{lt} failed: {exc}")
                continue

            params = result["params"]
            metrics = result["metrics"]
            result_rows.append(
                {
                    "model_scope": model_scope,
                    "leadtime": result["leadtime"],
                    "p": params["p"],
                    "d": params["d"],
                    "q": params["q"],
                    "P": params["P"],
                    "D": params["D"],
                    "Q": params["Q"],
                    "s": params["s"],
                    "seasonal_order": str(result["seasonal_order"]),
                    "best_trial_number": result["best_trial_number"],
                    "best_validation_RMSE": result["best_validation_RMSE"],
                    "MSE": metrics["MSE"],
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "MAPE": metrics["MAPE"],
                    "R2": metrics["R2"],
                    "n_train": result["n_train"],
                    "n_test": result["n_test"],
                    "exog_columns": ",".join(result["exog_columns"]),
                    "model_path": result["model_path"],
                    "n_trials": n_trials,
                    "maxiter": maxiter,
                }
            )
            prediction_frames.append(result["predictions"])
            pd.DataFrame(result_rows).sort_values("leadtime", ascending=False).to_csv(
                partial_path,
                index=False,
                encoding="utf-8-sig",
            )
            print(
                f"{model_scope} LT{result['leadtime']} done: "
                f"order=({params['p']},{params['d']},{params['q']}), "
                f"seasonal_order={result['seasonal_order']}, "
                f"RMSE={metrics['RMSE']:.4f}"
            )

    results_df = pd.DataFrame(result_rows).sort_values("leadtime", ascending=False).reset_index(drop=True)
    predictions_df = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    return results_df, predictions_df
