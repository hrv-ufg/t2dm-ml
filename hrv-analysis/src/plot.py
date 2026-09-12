import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import statsmodels.api as sm


# Função para plot estilo corrplot do R
def corrplot(
    corr,
    size_scale=500,
    savepath="../results/corrplot.png",
    tick_fontsize=12,
    figsize=(10, 8),
):
    cmap = plt.cm.RdBu

    fig, ax = plt.subplots(figsize=figsize)
    # desenha bordas quadradas para cada célula
    for i in range(len(corr)):
        for j in range(len(corr)):
            # quadrado com borda cinza clara
            rect = patches.Rectangle(
                (j - 0.5, -i - 0.5),
                1,
                1,
                linewidth=0.5,
                edgecolor="lightgray",
                facecolor="none",
            )
            ax.add_patch(rect)

            # agora os círculos
            c = corr.iloc[i, j]
            ax.scatter(
                j,
                -i,
                s=size_scale * abs(c),
                c=[[cmap((c + 1) / 2)]],
                alpha=0.9,
                marker="o",
            )

    # Ajustes de eixo
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=tick_fontsize)
    ax.set_yticks(-np.arange(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=tick_fontsize)
    ax.tick_params(axis="both", length=0)

    # remover os traços (ticks) dos eixos
    ax.tick_params(axis="both", length=0)

    # mover x para cima
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Ajustar limites para remover espaço em branco
    ax.set_xlim(-0.7, len(corr.columns) - 0.4)
    ax.set_ylim(-len(corr.index) + 0.4, 0.7)

    # Remover borda preta externa
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        ticks=np.arange(-1, 1.1, 0.2),
    )
    cbar.ax.tick_params(labelsize=tick_fontsize)  # aumenta fonte dos ticks da colorbar

    plt.tight_layout()

    # Criar pasta e salvar
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()