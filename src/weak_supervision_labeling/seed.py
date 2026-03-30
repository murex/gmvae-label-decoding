# scr/weak_supervision_labeling/seed.py

import random
import os
import torch
import numpy as np


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id):
    worker_seed = 0 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)