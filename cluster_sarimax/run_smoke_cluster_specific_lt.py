from full_pipeline_cluster_specific_lt_parallel_v1 import (
    ClusterSpecificConfig,
    run_cluster_specific_pipeline,
)


if __name__ == "__main__":
    run_cluster_specific_pipeline(
        ClusterSpecificConfig(
            output_dir="results/smoke_cluster_specific_lt_sarimax_parallel_v1",
            n_optuna_trials=1,
            n_jobs=2,
            maxiter=20,
            save_models=False,
            save_plots=False,
            leadtimes=[90, 0],
            run_label="smoke",
        )
    )
