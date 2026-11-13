from full_pipeline_cluster_specific_lt_parallel_v3 import (
    ClusterSpecificConfig,
    run_cluster_specific_pipeline,
)


if __name__ == "__main__":
    run_cluster_specific_pipeline(
        ClusterSpecificConfig(
            output_dir="results/smoke_cluster_specific_lt_sarimax_parallel_v3",
            n_optuna_trials=2,
            n_jobs=2,
            maxiter=20,
            save_models=False,
            save_plots=False,
            leadtimes=[90, 0],
            run_label="smoke_auto_s",
            tune_seasonal=True,
            seasonal_period_candidates=None,
            max_auto_seasonal_period=30,
            max_auto_seasonal_candidates=5,
        )
    )
