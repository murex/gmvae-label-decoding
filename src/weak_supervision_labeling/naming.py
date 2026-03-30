# scr/weak_supervision_labeling/naming.py


def method_family(method) -> str:
    name = str(getattr(method, "name", type(method).__name__)).lower().strip()
    if "gmvae" in name:
        return "gmvae"
    if "gmm" in name or "gaussianmixture" in name:
        return "gmm"
    return name.replace("/", "_").replace(" ", "_")


def latent_bucket(method) -> str:
    cfg = getattr(method, "cfg", {}) if isinstance(getattr(method, "cfg", None), dict) else {}
    z = cfg.get("z_dim", None) or cfg.get("z", None) or cfg.get("latent_dim", None)
    if z is None:
        return "zNA"
    try:
        return f"z{int(z)}"
    except Exception:
        return "zNA"


def method_tag(method, *, seed: int) -> str:
    name = str(getattr(method, "name", type(method).__name__)).lower().strip()
    cfg = getattr(method, "cfg", {}) if isinstance(getattr(method, "cfg", None), dict) else {}

    epochs = cfg.get("epochs", None)
    bs = cfg.get("batch_size", None)

    is_gmvae = "gmvae" in name
    is_vae = (("vae" in name) and not is_gmvae) or ("cluster_head" in cfg)

    parts = [name]

    if is_vae:
        z_dim = cfg.get("z_dim", None)
        hidden_dim = cfg.get("hidden_dim", None)
        beta = cfg.get("beta", None)
        head = cfg.get("cluster_head", None)
        n_clusters = cfg.get("n_clusters", None)

        if z_dim is not None:
            parts.append(f"z{int(z_dim)}")
        if hidden_dim is not None:
            parts.append(f"h{int(hidden_dim)}")
        if beta is not None:
            parts.append(f"beta{float(beta):g}")

        if head is None:
            parts.append("headnone")
        else:
            head = str(head).lower().strip()
            if head in {"gmm", "gaussianmixture"}:
                if n_clusters is not None:
                    parts.append(f"K{int(n_clusters)}")
                parts.append("gmm" if head == "gaussianmixture" else head)
            else:
                parts.append(head.replace("/", "_").replace(" ", "_"))

        if epochs is not None:
            parts.append(f"ep{int(epochs)}")
        if bs is not None:
            parts.append(f"bs{int(bs)}")
        parts.append(f"seed{int(seed)}")
        return "_".join(parts)

    z_dim = cfg.get("z_dim", None)
    K = cfg.get("K", None)
    hidden_dim = cfg.get("hidden_dim", None)
    n_layers = cfg.get("n_layers", None)

    if z_dim is not None:
        parts.append(f"z{int(z_dim)}")
    if K is not None:
        parts.append(f"K{int(K)}")
    if hidden_dim is not None:
        parts.append(f"h{int(hidden_dim)}")
    if n_layers is not None:
        parts.append(f"L{int(n_layers)}")
    if epochs is not None:
        parts.append(f"ep{int(epochs)}")
    if bs is not None:
        parts.append(f"bs{int(bs)}")
    parts.append(f"seed{int(seed)}")
    return "_".join(parts)