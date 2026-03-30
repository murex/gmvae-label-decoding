from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Titles:
    title: str = ""
    suptitle: str = ""

def save_close(fig, savepath: str | Path | None, *, dpi=300):
    if savepath is not None:
        fig.savefig(str(savepath), dpi=dpi, bbox_inches="tight")
    plt.close(fig)