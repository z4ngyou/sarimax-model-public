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
DEFAULT_MAX_AUTO_SEASONAL_PERIOD = 60
DEFAULT_MAX_AUTO_SEASONAL_CANDIDATES = 8


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


def calculate_information_criteria(model_results, n_obs: int) -> dict[str, float]:
    aic = float(getattr(model_results, "aic", np.nan))
    bic = float(getattr(model_results, "bic", np.nan))
    hqic = float(getattr(model_results, "hqic", np.nan))
    k_params = len(getattr(model_results, "params", []))
    if np.isfinite(aic) and n_obs > k_params + 1:
        aicc = aic + (2 * k_params * (k_params + 1)) / (n_obs - k_params - 1)
    else:
        aicc = np.nan
    return {
        "AIC": aic,
        "BIC": bic,
        "AICc": float(aicc),
        "HQIC": hqic,
        "k_params": int(k_params),
    }


def is_model_converged(model_results) -> bool:
    mle_retvals = getattr(model_results, "mle_retvals", {}) or {}
    return bool(mle_retvals.get("converged", False))


def _with_label(file_name: str, run_label: str | None = None) -> str:
    return f"{run_label}_{file_name}" if run_label else file_name


def _filter_seasonal_period_candidates(
    candidates: list[int] | None,
    n_train: int,
) -> list[int]:
    raw_candidates = candidates or [0]
    normalized = {0}
    for candidate in raw_candidates:
        s = int(candidate)
        if s == 0:
            normalized.add(0)
        elif s > 1 and n_train >= (2 * s + 5):
            normalized.add(s)
    return sorted(normalized)


def _auto_seasonal_period_candidates(
    train_data: pd.DataFrame,
    target_column: str,
    max_period: int,
    max_candidates: int,
) -> list[int]:
    y = (
        train_data.sort_values("date_part")[target_column]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy()
    )
    n = len(y)
    max_period = int(min(max_period, max(2, n // 3)))
    if n < 20 or max_period < 2 or np.nanstd(y) == 0:
        return [0]

    centered = y - np.nanmean(y)
    scores: dict[int, float] = {}

    for lag in range(2, max_period + 1):
        left = centered[:-lag]
        right = centered[lag:]
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom == 0:
            continue
        corr = float(np.dot(left, right) / denom)
        if corr > 0:
            scores[lag] = max(scores.get(lag, 0.0), corr)

    fft_values = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(n, d=1.0)
    powers = np.abs(fft_values) ** 2
    total_power = float(np.sum(powers[1:]))
    if total_power > 0:
        for freq, power in zip(freqs[1:], powers[1:]):
            if freq <= 0:
                continue
            period = int(round(1 / freq))
            if 2 <= period <= max_period:
                score = float(power / total_power)
                scores[period] = max(scores.get(period, 0.0), score)

    selected = [
        period
        for period, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:max_candidates]
    ]
    return [0] + sorted(selected)


def _resolve_seasonal_period_candidates(
    train_data: pd.DataFrame,
    target_column: str,
    explicit_candidates: list[int] | None,
    max_auto_period: int,
    max_auto_candidates: int,
) -> list[int]:
    if explicit_candidates is not None:
        return _filter_seasonal_period_candidates(explicit_candidates, len(train_data))
    auto_candidates = _auto_seasonal_period_candidates(
        train_data,
        target_column,
        max_auto_period,
        max_auto_candidates,
    )
    return _filter_seasonal_period_candidates(auto_candidates, len(train_data))


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
    validation_data,
    target_column,
    maxiter,
    tune_seasonal: bool,
    seasonal_period_candidates: list[int],
):
    p = trial.suggest_int("p", 0, 5)
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 5)

    if p + d + q > 9:
        return float("inf")

    P = D = Q = s = 0
    if tune_seasonal:
        s = trial.suggest_categorical("s", seasonal_period_candidates)
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
            len(validation_data),
            validation_data,
            target_column,
        )
        mse = mean_squared_error(test_data_indexed[target_column], forecast_mean)
        return math.sqrt(mse)
    except Exception:
        return float("inf")


def optimize_sarimax_parameters(
    train_data,
    validation_data,
    target_column,
    n_trials=30,
    maxiter: int = 50,
    tune_seasonal: bool = True,
    seasonal_period_candidates: list[int] | None = None,
    max_auto_seasonal_period: int = DEFAULT_MAX_AUTO_SEASONAL_PERIOD,
    max_auto_seasonal_candidates: int = DEFAULT_MAX_AUTO_SEASONAL_CANDIDATES,
):
    resolved_seasonal_candidates = _resolve_seasonal_period_candidates(
        train_data,
        target_column,
        seasonal_period_candidates,
        max_auto_seasonal_period,
        max_auto_seasonal_candidates,
    )
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(
            trial,
            train_data,
            validation_data,
            target_column,
            maxiter,
            tune_seasonal,
            resolved_seasonal_candidates,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    if study.best_value == float("inf"):
        return {"p": 1, "d": 1, "q": 1, "P": 0, "D": 0, "Q": 0, "s": 0}, float("inf"), None, resolved_seasonal_candidates

    params = _complete_params(dict(study.best_params))
    return params, study.best_value, study.best_trial.number, resolved_seasonal_candidates


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
        max_auto_seasonal_period,
        max_auto_seasonal_candidates,
    ) = task
    train_data = data["train"]
    validation_data = data.get("validation")
    test_data = data["test"]
    if validation_data is None or validation_data.empty:
        validation_data = test_data
        refit_data = train_data
        validation_strategy = "fallback_test_as_validation"
    else:
        refit_data = pd.concat([train_data, validation_data], ignore_index=True).sort_values("date_part")
        validation_strategy = "holdout_validation"
    exog_columns = get_exog_columns(train_data)

    best_params, best_validation_rmse, best_trial_number, resolved_seasonal_candidates = optimize_sarimax_parameters(
        train_data,
        validation_data,
        target_column,
        n_trials=n_trials,
        maxiter=maxiter,
        tune_seasonal=tune_seasonal,
        seasonal_period_candidates=seasonal_period_candidates,
        max_auto_seasonal_period=max_auto_seasonal_period,
        max_auto_seasonal_candidates=max_auto_seasonal_candidates,
    )
    order = (best_params["p"], best_params["d"], best_params["q"])
    seasonal_order = _seasonal_order_from_params(best_params)

    validation_model_results = fit_sarimax_model(
        train_data,
        target_column,
        order=order,
        seasonal_order=seasonal_order,
        maxiter=maxiter,
    )
    validation_forecast_mean, _, validation_data_indexed = make_predictions(
        validation_model_results,
        len(validation_data),
        validation_data,
        target_column,
    )
    validation_metrics = evaluate_model(validation_data_indexed[target_column], validation_forecast_mean)

    model_results = fit_sarimax_model(
        refit_data,
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
    information_criteria = calculate_information_criteria(model_results, len(refit_data))
    converged = is_model_converged(model_results)

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
            "evaluation_split": "test",
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
        "validation_metrics": validation_metrics,
        "predictions": predictions_df,
        "seasonal_order": seasonal_order,
        "seasonal_period_candidates": resolved_seasonal_candidates,
        "information_criteria": information_criteria,
        "converged": converged,
        "validation_strategy": validation_strategy,
        "exog_columns": exog_columns,
        "model_path": model_path,
        "n_train": len(train_data),
        "n_validation": len(validation_data),
        "n_refit": len(refit_data),
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
    max_auto_seasonal_period: int = DEFAULT_MAX_AUTO_SEASONAL_PERIOD,
    max_auto_seasonal_candidates: int = DEFAULT_MAX_AUTO_SEASONAL_CANDIDATES,
):
    os.makedirs(output_dir, exist_ok=True)
    requested_leadtimes = set(leadtimes) if leadtimes is not None else None
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
            max_auto_seasonal_period,
            max_auto_seasonal_candidates,
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
            validation_metrics = result["validation_metrics"]
            information_criteria = result["information_criteria"]
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
                    "seasonal_period_candidates": ",".join(str(s) for s in result["seasonal_period_candidates"]),
                    "best_trial_number": result["best_trial_number"],
                    "best_validation_RMSE": result["best_validation_RMSE"],
                    "validation_MSE": validation_metrics["MSE"],
                    "validation_RMSE": validation_metrics["RMSE"],
                    "validation_MAE": validation_metrics["MAE"],
                    "validation_MAPE": validation_metrics["MAPE"],
                    "validation_R2": validation_metrics["R2"],
                    "MSE": metrics["MSE"],
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "MAPE": metrics["MAPE"],
                    "R2": metrics["R2"],
                    "AIC": information_criteria["AIC"],
                    "BIC": information_criteria["BIC"],
                    "AICc": information_criteria["AICc"],
                    "HQIC": information_criteria["HQIC"],
                    "k_params": information_criteria["k_params"],
                    "converged": result["converged"],
                    "validation_strategy": result["validation_strategy"],
                    "n_train": result["n_train"],
                    "n_validation": result["n_validation"],
                    "n_refit": result["n_refit"],
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
                f"s_candidates={result['seasonal_period_candidates']}, "
                f"validation_RMSE={validation_metrics['RMSE']:.4f}, "
                f"test_RMSE={metrics['RMSE']:.4f}"
            )

    results_df = pd.DataFrame(result_rows).sort_values("leadtime", ascending=False).reset_index(drop=True)
    predictions_df = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    return results_df, predictions_df
