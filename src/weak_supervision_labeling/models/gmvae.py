# src/weak_supervision_labeling/gmvae.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
from tqdm import tqdm
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from weak_supervision_labeling.helpers import make_loader_from_numpy, kl_diag_gaussians
from weak_supervision_labeling.io import save_json, ensure_dir


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2, act=nn.ReLU):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden_dim), act()]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GMVAE(nn.Module):
    def __init__(self, x_dim=784, z_dim=10, K=10, hidden_dim=500, n_layers=2):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.K = K

        self.encoder = MLP(x_dim, hidden_dim, hidden_dim, n_layers=n_layers)

        # q(c|x)
        self.qc = nn.Linear(hidden_dim, self.K)

        # q(z|x,c) : K heads
        self.qz_mu = nn.Linear(hidden_dim, self.K * z_dim)
        self.qz_logvar = nn.Linear(hidden_dim, self.K * z_dim)

        # p(x|z)
        # self.decoder = nn.Sequential(
        #     nn.Linear(z_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, x_dim),
        #     nn.Sigmoid()
        # )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512), nn.ReLU(),
            nn.Linear(512, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, x_dim), nn.Sigmoid(),
        )

        # p(c) = Cat(pi) (uniforme)
        self.register_buffer("log_pi", torch.log(torch.ones(K) / K))

        # p(z|c) = N(mu_c, diag(var_c))
        self.pz_mu = nn.Parameter(torch.randn(self.K, z_dim) * 0.05)
        self.pz_logvar = nn.Parameter(torch.zeros(self.K, z_dim))

    def encode(self, x):
        h = self.encoder(x)                 # (B, H)
        qc_logits = self.qc(h)              # (B, K)
        qc = F.softmax(qc_logits, dim=-1)   # (B, K)

        mu_all = self.qz_mu(h).view(-1, self.K, self.z_dim)         # (B, K, D)
        logvar_all = self.qz_logvar(h).view(-1, self.K, self.z_dim) # (B, K, D)
        return qc_logits, qc, mu_all, logvar_all

    def reparam(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z):
        return self.decoder(z)

    def elbo(self, x, recon_loss="bce"):
        """
        x: (B, x_dim) in [0,1]
        returns: scalar loss, and dict diagnostics
        """
        B = x.size(0)
        qc_logits, qc, mu_all, logvar_all = self.encode(x)

        # sample z for each head
        z_all = self.reparam(mu_all, logvar_all)  # (B,K,D)

        # decode each z(x,c)
        x_hat_all = self.decode(z_all.reshape(B * self.K, self.z_dim))  # (B*K, x_dim)
        x_rep = x.unsqueeze(1).expand(B, self.K, self.x_dim).reshape(B * self.K, self.x_dim)

        if recon_loss == "bce":
            recon = F.binary_cross_entropy(x_hat_all, x_rep, reduction="none").sum(dim=-1)
        else:
            recon = F.mse_loss(x_hat_all, x_rep, reduction="none").sum(dim=-1)
        recon = recon.view(B, self.K)

        # KL(q(z|x,c) || p(z|c))
        mu_p = self.pz_mu.unsqueeze(0).expand(B, self.K, self.z_dim)
        logvar_p = self.pz_logvar.unsqueeze(0).expand(B, self.K, self.z_dim)
        kl_z = kl_diag_gaussians(mu_all, logvar_all, mu_p, logvar_p)

        # log p(c) and log q(c|x)
        log_pc = self.log_pi.unsqueeze(0).expand(B, self.K)
        log_qc = F.log_softmax(qc_logits, dim=-1)

        # negative ELBO
        loss_per_x = torch.sum(qc * (recon + kl_z + log_qc - log_pc), dim=-1)
        loss = loss_per_x.mean()

        diag = {
            "reconstruction": (qc * recon).sum(dim=-1).mean().item(),
            "KL_z": (qc * kl_z).sum(dim=-1).mean().item(),
            "H(C|X)": (-torch.sum(qc * log_qc, dim=-1)).mean().item(),
        }
        return loss, diag

    @torch.no_grad()
    def sample_from_prior(self, n):
        pi = torch.softmax(self.log_pi, dim=-1)
        c = torch.multinomial(pi, num_samples=n, replacement=True)  # (n,)
        mu = self.pz_mu[c]
        logvar = self.pz_logvar[c]
        z = self.reparam(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, c
        


# ------------------------------------------------------------
# Method wrapper
# ------------------------------------------------------------


class GMVAEMethod:
    """
    Wrapper "method" pour ton framework.
    - fit(X) entraîne le GMVAE
    - predict(X, mode=...) renvoie des clusters selon le mode
    - predict_proba(X) renvoie q(c|x)
    - embed(X) renvoie un embedding latent (voir embedding_mode)
    """

    name = "gmvae"

    def __init__(
        self,
        x_dim: int = 784,
        z_dim: int = 10,
        K: int = 10,
        hidden_dim: int = 200,
        n_layers: int = 2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 128,
        recon_loss: str = "bce",  # "bce" or "mse"
        embedding_mode: str = "expected_mu",  # "expected_mu" | "mode_mu"
        device: Optional[str] = None,
        num_workers: int = 0,
        verbose: bool = True,
        dataset: str | None = None,
        tb_log_images_every: int = 500,
        tb_log_hist_every: int = 200,
    ):
        recon_loss = str(recon_loss).lower().strip()
        if recon_loss not in {"bce", "mse"}:
            raise ValueError("recon_loss must be 'bce' or 'mse'")

        embedding_mode = str(embedding_mode).lower().strip()
        if embedding_mode not in {"expected_mu", "mode_mu"}:
            raise ValueError("embedding_mode must be 'expected_mu' or 'mode_mu'")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cfg: Dict[str, Any] = dict(
            x_dim=int(x_dim),
            z_dim=int(z_dim),
            K=int(K),
            hidden_dim=int(hidden_dim),
            n_layers=int(n_layers),
            lr=float(lr),
            epochs=int(epochs),
            batch_size=int(batch_size),
            recon_loss=recon_loss,
            embedding_mode=embedding_mode,
            num_workers=int(num_workers),
            verbose=bool(verbose),
            dataset=(dataset.lower().strip() if isinstance(dataset, str) else dataset),
            tb_log_images_every=int(tb_log_images_every),
            tb_log_hist_every=int(tb_log_hist_every),
        )

        self.device = torch.device(device)

        self.model = GMVAE(
            x_dim=int(x_dim),
            z_dim=int(z_dim),
            K=int(K),
            hidden_dim=int(hidden_dim),
            n_layers=int(n_layers),
        ).to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(lr))

        self.history: Dict[str, list[float]] = {
            "loss": [],
            "reconstruction": [],
            "KL_z": [],
            "H(C|X)": [],
        }

        # -------- Plot strings / titles (kept as you had) --------
        self.name_plot = f"GMVAE ($K = {K}$)"
        self.cm_title = (
            r"GMVAE native clusters ($c_x = \mathrm{arg\,max}\,q(c\mid x)$)"
            + f"  ($K = {K}$)"
            "\nConfusion Matrix"
        )

        self.plot_suptitle_pca = r"GMVAE mean latent state $\mu_{c_x}$ colored by true $y$ (PCA)"
        self.plot_suptitle_tsne = r"GMVAE mean latent state $\mu_{c_x}$ colored by true $y$ (t-SNE)"
        self.plot_suptitle_pca_pred = r"GMVAE mean latent state $\mu_{c_x}$ colored by clusters $c_x$"
        self.plot_suptitle_tsne_pred = r"GMVAE mean latent state $\mu_{c_x}$ colored by clusters $c_x$"

        self.plot_title = r"$c_x\sim Cat(q(c\mid x))$ and $z\sim\mathcal{N}(\mu_{c_x}, \Sigma_{c_x})$"

        self.plot_generated_title = (
            r"Generated samples per cluster from GMVAE"
            + f"  ($K = {K}$)"
            "\n"
            r"$z\sim p(z|c)$ and $x\sim p(x|z)$ for $c=1...K$"
        )

        self.plot_generated_title_mu = (
            r"Generated samples per cluster"
            "\n"
            r"$z_c = \mu_c$ with $z\sim p(z\mid c)$ and $x\sim p(x\mid z)$, for $c=1,\dots,K$"
        )

        self.plot_generated_within_cluster_title = (
            r"Generated samples within a fixed GMVAE cluster"
            + f"  ($K = {K}$)"
            "\n"
            r"$c$ fixed, $z \sim p(z \mid c)$ and $x \sim p(x \mid z)$"
        )



    def _loader(self, X: np.ndarray, *, shuffle: bool, batch_size: int | None = None):
        if batch_size is None:
            batch_size = int(self.cfg["batch_size"])
        return make_loader_from_numpy(
            X,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=int(self.cfg.get("num_workers", 0)),
        )

    # -------------------------
    # Training
    # -------------------------
    def fit(self, X: np.ndarray, y: np.ndarray | None = None, writer=None) -> "GMVAEMethod":
        """
        Train GMVAE on X (labels y unused). Logs to TensorBoard if writer is provided.
        The caller (main) is responsible for writer.flush()/writer.close().
        """
        X = np.asarray(X, dtype=np.float32)
        self.model.train()

        recon_loss = str(self.cfg.get("recon_loss", "bce"))
        epochs = int(self.cfg["epochs"])
        verbose = bool(self.cfg.get("verbose", True))
        K = int(self.cfg["K"])

        loader = self._loader(X, shuffle=True)

        if not hasattr(self, "_tb_step"):
            self._tb_step = 0

        log_images_every = int(self.cfg.get("tb_log_images_every", 500))
        log_hist_every = int(self.cfg.get("tb_log_hist_every", 200))

        t0 = time.perf_counter()

        for ep in range(epochs):
            ep_loss = 0.0
            ep_rec = 0.0
            ep_kl = 0.0
            ep_h = 0.0
            n_batches = 0

            it = loader
            if verbose:
                it = tqdm(loader, desc=f"GMVAE epoch {ep+1}/{epochs}", leave=False)

            for (xb,) in it:
                xb = xb.to(self.device, non_blocking=True)

                loss, diag = self.model.elbo(xb, recon_loss=recon_loss)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()

                ep_loss += float(loss.item())
                ep_rec += float(diag.get("reconstruction", 0.0))
                ep_kl += float(diag.get("KL_z", 0.0))
                ep_h += float(diag.get("H(C|X)", 0.0))
                n_batches += 1

                if writer is not None:
                    gs = int(self._tb_step)

                    writer.add_scalar("train/loss", float(loss.item()), gs)
                    writer.add_scalar("train/reconstruction", float(diag.get("reconstruction", 0.0)), gs)
                    writer.add_scalar("train/KL_z", float(diag.get("KL_z", 0.0)), gs)
                    writer.add_scalar("train/H_C_given_X", float(diag.get("H(C|X)", 0.0)), gs)

                    with torch.no_grad():
                        # encode returns qc already (softmax)
                        _, qc, _, _ = self.model.encode(xb)  # qc: (B,K)
                        qbar = qc.mean(dim=0)  # (K,)

                        usage_entropy = -(qbar * (qbar.clamp_min(1e-12)).log()).sum()
                        log_unif = -torch.log(torch.tensor(float(K), device=qc.device))
                        usage_kl_to_unif = (qbar * ((qbar.clamp_min(1e-12)).log() - log_unif)).sum()

                        mean_max_qc = qc.max(dim=1).values.mean()
                        h_c_given_x = -(qc * (qc.clamp_min(1e-12)).log()).sum(dim=1).mean()

                    writer.add_scalar("train/usage_entropy", float(usage_entropy.item()), gs)
                    writer.add_scalar("train/usage_kl_to_unif", float(usage_kl_to_unif.item()), gs)
                    writer.add_scalar("train/mean_max_qc", float(mean_max_qc.item()), gs)
                    writer.add_scalar("train/H_C_given_X_batch", float(h_c_given_x.item()), gs)

                    if (log_hist_every > 0) and (gs % log_hist_every == 0):
                        writer.add_histogram("train/qbar", qbar.detach().cpu().numpy(), gs)

                    # Images only if it makes sense (MNIST-like)
                    if (log_images_every > 0) and (gs % log_images_every == 0) and int(self.cfg["x_dim"]) == 784:
                        with torch.no_grad():
                            x0 = xb[:16]
                            _, qc0, mu0, _ = self.model.encode(x0)
                            c0 = qc0.argmax(dim=1)
                            z0 = mu0[torch.arange(x0.size(0), device=x0.device), c0]
                            x_hat0 = self.model.decode(z0)

                            try:
                                writer.add_images("train/x", x0.view(-1, 1, 28, 28), gs)
                                writer.add_images("train/x_hat", x_hat0.view(-1, 1, 28, 28), gs)
                            except Exception:
                                pass

                            try:
                                xs, _ = self.model.sample_from_prior(16)
                                writer.add_images("train/prior_samples", xs.view(-1, 1, 28, 28), gs)
                            except Exception:
                                pass

                self._tb_step += 1

                if verbose:
                    it.set_postfix(
                        loss=f"{loss.item():.3f}",
                        rec=f"{diag.get('reconstruction', 0.0):.3f}",
                        kl=f"{diag.get('KL_z', 0.0):.3f}",
                        h=f"{diag.get('H(C|X)', 0.0):.3f}",
                    )

            if n_batches > 0:
                ep_loss /= n_batches
                ep_rec /= n_batches
                ep_kl /= n_batches
                ep_h /= n_batches

            self.history["loss"].append(float(ep_loss))
            self.history["reconstruction"].append(float(ep_rec))
            self.history["KL_z"].append(float(ep_kl))
            self.history["H(C|X)"].append(float(ep_h))

            if writer is not None:
                writer.add_scalar("epoch/loss", ep_loss, ep)
                writer.add_scalar("epoch/reconstruction", ep_rec, ep)
                writer.add_scalar("epoch/KL_z", ep_kl, ep)
                writer.add_scalar("epoch/H_C_given_X", ep_h, ep)

        self._train_time_sec = float(time.perf_counter() - t0)
        self.model.eval()
        return self

    def fit_predict(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        gmvae_cluster_mode: str = "qc",  # "qc"
        reps: dict[int, Any] | None = None,
        qc_sample_seed: int = 0,
        batch_size: int | None = None,
        writer=None,
    ) -> np.ndarray:
        self.fit(X, y, writer=writer)

        m = str(gmvae_cluster_mode).lower().strip()
        if m == "qc":
            mode = "qc_argmax"
        else:
            raise ValueError(f"Unknown gmvae_cluster_mode={gmvae_cluster_mode!r} (expected qc)")

        return self.predict(
            X,
            mode=mode,
            reps=reps,
            qc_sample_seed=int(qc_sample_seed),
            batch_size=batch_size,
        )


    @torch.no_grad()
    def predict(
        self,
        X: np.ndarray,
        *,
        mode: str = "qc_argmax",  # "qc_argmax"
        batch_size: int | None = None,
        qc_sample_seed: int = 0,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)

        mode = str(mode).lower().strip()
        if mode not in {"qc_argmax"}:
            raise ValueError(f"Unknown predict mode={mode}")


        def _qc():
            qc = np.asarray(self.predict_proba(X, batch_size=batch_size), dtype=np.float64)
            if qc.ndim != 2:
                raise ValueError(f"predict_proba must return (N,K), got {qc.shape}")
            qc = np.clip(qc, 0.0, 1.0)
            qc = qc / np.maximum(qc.sum(axis=1, keepdims=True), 1e-12)
            return qc

        if mode == "qc_argmax":
            qc = _qc()
            return qc.argmax(axis=1).astype(int)


    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, *, batch_size: int | None = None) -> np.ndarray:
        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        loader = self._loader(X, shuffle=False, batch_size=batch_size)

        probas = []
        for (xb,) in loader:
            xb = xb.to(self.device, non_blocking=True)
            _, qc, _, _ = self.model.encode(xb)
            probas.append(qc.detach().cpu().numpy())
        return np.concatenate(probas, axis=0)

    @torch.no_grad()
    def predict_argmax(self, X: np.ndarray) -> np.ndarray:
        qc = self.predict_proba(X)
        return qc.argmax(axis=1).astype(int)

    @torch.no_grad()
    def embed(self, X: np.ndarray, *, batch_size: int | None = None) -> np.ndarray:
        """
        Embedding latent.
        - expected_mu: E_q[c|x][ mu(x,c) ] = sum_k q(c=k|x) * mu_k(x)
        - mode_mu: mu(x, argmax q(c|x))
        """
        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        loader = self._loader(X, shuffle=False, batch_size=batch_size)

        embs = []
        mode = str(self.cfg["embedding_mode"]).lower().strip()

        for (xb,) in loader:
            xb = xb.to(self.device, non_blocking=True)
            _, qc, mu_all, _ = self.model.encode(xb)  # qc:(B,K), mu_all:(B,K,D)

            if mode == "expected_mu":
                z = torch.sum(qc.unsqueeze(-1) * mu_all, dim=1)  # (B,D)
            elif mode == "mode_mu":
                c = torch.argmax(qc, dim=1)
                z = mu_all[torch.arange(mu_all.size(0), device=xb.device), c]
            else:
                raise ValueError(f"Unknown embedding_mode={self.cfg['embedding_mode']}")

            embs.append(z.detach().cpu().numpy())

        return np.concatenate(embs, axis=0)

    def extra(self) -> Dict[str, Any]:
        return {"history": self.history, "cfg": self.cfg, "device": str(self.device)}

    # Save / load
    def can_save(self) -> bool:
        return True

    def save(self, ckpt_dir: Path) -> None:
        ckpt_dir = ensure_dir(Path(ckpt_dir))
        save_json(self.cfg, ckpt_dir / "cfg.json")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "train_time_sec": getattr(self, "_train_time_sec", None),
                "history": getattr(self, "history", {}),
            },
            ckpt_dir / "model.pt",
        )

    def load(self, ckpt_dir: Path) -> bool:
        ckpt_dir = Path(ckpt_dir)
        f_model = ckpt_dir / "model.pt"
        if not f_model.exists():
            return False

        payload = torch.load(f_model, map_location=self.device)
        self._train_time_sec = payload.get("train_time_sec", None)
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.history = payload.get("history", {})
        return True

    def set_train_time(self, seconds: float | None):
        self._train_time_sec = None if seconds is None else float(seconds)

    def get_train_time(self) -> float | None:
        return getattr(self, "_train_time_sec", None)

    # Posterior params
    @torch.no_grad()
    def posterior_params(self, X: np.ndarray, batch_size: int = 256):
        """
        Returns (qc, mus, vars) with shapes:
          qc:   (N, K)
          mus:  (N, K, D)
          vars: (N, K, D)
        """
        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        loader = self._loader(X, shuffle=False, batch_size=int(batch_size))

        qc_list, mu_list, var_list = [], [], []
        for (xb,) in loader:
            xb = xb.to(self.device, non_blocking=True)
            _, qc, mu_all, logvar_all = self.model.encode(xb)
            qc_list.append(qc.detach().cpu().numpy())
            mu_list.append(mu_all.detach().cpu().numpy())
            var_list.append(np.exp(logvar_all.detach().cpu().numpy()))

        return (
            np.concatenate(qc_list, axis=0),
            np.concatenate(mu_list, axis=0),
            np.concatenate(var_list, axis=0),
        )

    # Reconstruction
    @torch.no_grad()
    def reconstruct(
        self,
        X: np.ndarray,
        *,
        mode: str = "mode_mu",
        sample: bool | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Reconstruct x -> x_hat using the GMVAE.

        Backward-compatible API:
        - if sample is provided:
            sample=True  => mode="sample"
            sample=False => keep mode
        - mode:
            "mode_mu"     (deterministic)
            "expected_mu" (deterministic)
            "sample"      (stochastic)
        """
        if sample is not None:
            mode = "sample" if sample else mode

        mode = str(mode).lower().strip()
        if mode not in {"mode_mu", "expected_mu", "sample"}:
            raise ValueError(f"Unknown reconstruct mode={mode}")

        self.model.eval()
        X = np.asarray(X, dtype=np.float32)

        if batch_size is None:
            batch_size = int(self.cfg.get("batch_size", 128))

        loader = self._loader(X, shuffle=False, batch_size=int(batch_size))

        outs = []
        for (xb,) in loader:
            xb = xb.to(self.device, non_blocking=True)

            _, qc, mu_all, logvar_all = self.model.encode(xb)  # qc:(B,K), mu/logvar:(B,K,D)
            B = mu_all.size(0)

            if mode == "mode_mu":
                c = torch.argmax(qc, dim=1)
                z = mu_all[torch.arange(B, device=xb.device), c]

            elif mode == "expected_mu":
                z = torch.sum(qc.unsqueeze(-1) * mu_all, dim=1)

            else:  # sample
                c = torch.multinomial(qc, num_samples=1).squeeze(1)
                mu = mu_all[torch.arange(B, device=xb.device), c]
                lv = logvar_all[torch.arange(B, device=xb.device), c]
                z = self.model.reparam(mu, lv)

            x_hat = self.model.decode(z)
            outs.append(x_hat.detach().cpu().numpy())

        return np.concatenate(outs, axis=0)