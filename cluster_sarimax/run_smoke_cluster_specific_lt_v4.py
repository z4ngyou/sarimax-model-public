from full_pipeline_cluster_specific_lt_parallel_v4 import (
    ClusterSpecificConfig,
    run_cluster_specific_pipeline,
)


if __name__ == "__main__":
    run_cluster_specific_pipeline(
        ClusterSpecificConfig(
            output_dir="results/smoke_cluster_specific_lt_sarimax_parallel_v4",
            n_optuna_trials=2,
            n_jobs=2,
            maxiter=20,
            save_models=False,
            save_plots=False,
            leadtimes=[90, 0],
            run_label="smoke_validation",
            tune_seasonal=True,
            seasonal_period_candidates=None,
            validation_fraction=0.2,
            max_auto_seasonal_period=30,
            max_auto_seasonal_candidates=5,
        )
    )
