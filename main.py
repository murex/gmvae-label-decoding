from run_setup import setup_env, setup_plot

setup_env()
setup_plot()

from weak_supervision_labeling.experiment import run


if __name__ == "__main__":
    # model seed
    seed = 0

    # number of training epochs
    n_epochs = 450

    # dataset
    dataset = "emnist" # available: "mnist", "emnist"

    # fraction of labeled data (between 0 and 1)
    label_map_frac_eval = 0.002

    # whether to display titles in plots
    titles_plot = True

    # whether to skip the UMAP representation of the latent space
    skip_umap = True

    # whether to skip the t-SNE representation of the latent space
    skip_tsne = True

    # supervised baselines to compare
    supervised_baselines = ["logreg", "mlp", "xgboost"]  # available: "logreg", "mlp", "xgboost"

    # number of seeds to compare baselines to the implemented method
    label_map_n_seeds = 5


    run(
        seed=seed,
        n_epochs=n_epochs,
        dataset=dataset,
        titles_plot=titles_plot,
        skip_umap=skip_umap,
        skip_tsne=skip_tsne,
        label_map_frac_eval=label_map_frac_eval,
        label_map_stratified=False,
        label_map_fracs=[5e-5, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 5e-1],
        supervised_baselines=supervised_baselines,
        label_map_n_seeds=label_map_n_seeds,
        save_gmvae_model=True,
    )