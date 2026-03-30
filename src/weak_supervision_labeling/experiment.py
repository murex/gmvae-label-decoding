# scr/weak_supervision_labeling/experiment.py

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence
import torch
from tensorboardX import SummaryWriter
from weak_supervision_labeling.data.mnist import load_mnist_flat
from weak_supervision_labeling.data.emnist import load_emnist_letters_flat
from weak_supervision_labeling.seed import seed_everything
from weak_supervision_labeling.io import ensure_dir, save_json
from weak_supervision_labeling.plotting.pipeline import PlotRunner, PlotConfig
from weak_supervision_labeling.helpers import print_metrics
from weak_supervision_labeling.naming import method_tag, method_family, latent_bucket
from weak_supervision_labeling.models.gmvae import GMVAEMethod
from weak_supervision_labeling.weak_supervision import prepare_decoding
from weak_supervision_labeling.paths import RUNS_DIR, CHECKPOINTS_DIR




def get_methods(device=None, n_epochs=4, dataset=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return [
        GMVAEMethod(
            z_dim=100,
            K=100,
            n_layers=4,
            hidden_dim=768,
            epochs=n_epochs,
            batch_size=128,
            device=device,
            dataset=dataset,
        ),
    ]


def make_writer(tb_dir: Path, enabled: bool) -> SummaryWriter | None:
    if not enabled:
        return None
    tb_dir = ensure_dir(tb_dir)
    return SummaryWriter(log_dir=str(tb_dir))


def run(
    seed=0,
    save_gmvae_model: bool = False,
    n_epochs: int = 4,
    dataset: str = "mnist",
    titles_plot: bool = True,
    skip_umap: bool = True,
    skip_tsne: bool = True,
    label_map_frac_eval: float = 0.10,
    label_map_stratified: bool = True,
    label_map_fracs: list[float] | None = None,
    supervised_baselines: Sequence[str] | None = ("logreg",),
    label_map_n_seeds: int | None = 5,
):
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = dataset.lower().strip()
    if dataset == "mnist":
        X, y = load_mnist_flat(normalize=True)
        print(f"MNIST: X={X.shape}")
    elif dataset == "emnist":
        X, y = load_emnist_letters_flat(normalize=True)
        print(f"EMNIST Letters: X={X.shape}")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    run_root = ensure_dir(RUNS_DIR / dataset / "plots")
    ckpt_root = ensure_dir(CHECKPOINTS_DIR / dataset)

    print(f"Run root: outputs/runs/{dataset}/plots")
    print(f"Checkpoint root: outputs/checkpoints/{dataset}")

    methods = get_methods(device=device, n_epochs=n_epochs, dataset=dataset)


    for method in methods:
        name = getattr(method, "name", type(method).__name__)

        tag = method_tag(method, seed=seed)
        family = method_family(method)
        zdir = latent_bucket(method)
        method_dir = ensure_dir(run_root / family / zdir / tag) #/home/lsaci/TDS_CODE/outputs/runs/mnist/plots
        results_dir = ensure_dir(method_dir / "results")
        ckpt_dir = ensure_dir(ckpt_root / tag)
        tb_dir = method_dir / "tb"


        

        # print("==================================================")
        # print(f"[{name.upper()}]")
        # print("==================================================")
        print()
        print(Path(f"Output dir: outputs/runs/{dataset}/plots") / family / zdir / tag)


        loaded = False
        if save_gmvae_model and callable(getattr(method, "can_save", None)) and method.can_save():
            try:
                loaded = method.load(ckpt_dir)
            except Exception as e:
                print(f"  -> load failed: {e}")
                loaded = False

        writer = make_writer(tb_dir, enabled=(not loaded))

        if loaded:
            print("==================================================")
            print(f"Loding the model (GMVAE)...")
            print("==================================================")
            print("  -> loaded model, skip training")
        else:
            print("==================================================")
            print(f"Training the model ({name.upper()})")
            print("==================================================")
            t0 = time.perf_counter()
            try:
                if callable(getattr(method, "fit", None)):
                    method.fit(X, y, writer=writer)
                else:
                    raise AttributeError(f"{type(method).__name__} has no fit().")
            finally:
                if writer is not None:
                    writer.flush()
                    writer.close()
            
            print()
            print(f"  -> training time: {time.perf_counter() - t0:.2f}s")
            print()

            if save_gmvae_model and callable(getattr(method, "can_save", None)) and method.can_save():
                try:
                    method.save(ckpt_dir)
                    print("  -> saved model")
                except Exception as e:
                    print(f"  -> save failed: {e}")


        modes = ["decoding_soft", "decoding_hard"]

        print("  preparing decoding cache...")
        decoding_cache = prepare_decoding(method, label_map_frac_eval, label_map_stratified, seed=seed, X=X, y=y)

        count = 0

        for mode in modes:
            print(f"  [EVAL] {mode}")

            if mode == "decoding_soft":
                if decoding_cache is None:
                    raise RuntimeError("decoding_cache is None for decoding_soft")
                y_pred = decoding_cache["y_pred_soft"]

            elif mode == "decoding_hard":
                if decoding_cache is None:
                    raise RuntimeError("decoding_cache is None for decoding_hard")
                y_pred = decoding_cache["y_pred_hard"]

            else:
                raise ValueError(f"Unknown eval mode={mode}")

            m_train = print_metrics(f"  Metrics", y, y_pred)
            save_json(m_train, results_dir / f"{mode}.json")

            if count==0:
                runner = PlotRunner(dataset=dataset)
                cfg_plot = PlotConfig(
                    seed=seed,
                    titles_plot=titles_plot,
                    skip_umap=skip_umap,
                    skip_tsne=skip_tsne,
                )
                cfg_plot.mode_name = mode
                cfg_plot.label_map_frac_eval = float(label_map_frac_eval)
                cfg_plot.label_map_stratified = bool(label_map_stratified)
                cfg_plot.label_map_fracs = label_map_fracs
                cfg_plot.supervised_baselines = supervised_baselines
                
                cfg_plot.label_map_n_seeds = label_map_n_seeds
                cfg_plot.label_map_seeds = list(range(label_map_n_seeds))

                runner.run(
                        method=method,
                        method_dir=method_dir,
                        decoding_cache=decoding_cache,
                        ckpt_dir=ckpt_dir,
                        X=X,
                        y=y,
                        y_pred=y_pred,
                        cfg=cfg_plot,
                    )
                count += 1
            print()