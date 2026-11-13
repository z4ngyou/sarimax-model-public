from full_pipeline_cluster_specific_lt_parallel_v4 import (
    ClusterSpecificConfig,
    run_cluster_specific_pipeline,
)


if __name__ == "__main__":
    run_cluster_specific_pipeline(
        ClusterSpecificConfig(
            output_dir="results/cluster_only_lt_sarimax_parallel_v4",
            n_optuna_trials=100,
            n_jobs=8,
            maxiter=50,
            save_models=True,
            save_plots=False,
            include_all_normal=False,
            leadtimes=None,
            run_label="cluster_only_2025dec_100trial_validation",
            tune_seasonal=True,
            seasonal_period_candidates=None,
            validation_fraction=0.2,
            max_auto_seasonal_period=60,
            max_auto_seasonal_candidates=8,
        )
    )
