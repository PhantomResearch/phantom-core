from matplotlib.figure import Figure
import warnings
import matplotlib.pyplot as plt


def fig_is_tight(fig: Figure) -> bool:

    with warnings.catch_warnings(record=True) as w:
        fig.tight_layout()
        if w:
            # plt.close(fig=fig)
            return False
    return True


DIM_MAP = {
    'width': 0,
    'height': 1,
}

def optimize_size(fig, which: str, delta: float = 1.0) -> Figure:

    dim = DIM_MAP[which]

    while not fig_is_tight(fig=fig):
        
        new_size = fig.get_size_inches()

        new_size[dim] += delta

        if new_size[dim] > 30:
            break

        fig.set_size_inches(new_size)

    return fig

