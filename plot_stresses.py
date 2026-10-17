import io
import contextlib
import runpy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize

# Grafica sigma_1 y sigma_2 sobre la configuracion deformada de la presa.
# Corre dam.py (con el ELEMENT, BASE_BC y malla que tenga configurados) y usa sus resultados.

SCALE = 1000   # Factor de escala de la deformada

with contextlib.redirect_stdout(io.StringIO()):
    dam = runpy.run_path("dam.py")
elements = dam["elements"]
u = dam["u"]
H, b, c = dam["H"], dam["b"], dam["c"]

# Orden de los nodos para dibujar el contorno de cada elemento (esquinas y lados intercalados)
OUTLINE = {"TRI3": [0, 1, 2], "TRI6": [0, 3, 1, 4, 2, 5],
           "QUAD4": [0, 1, 2, 3], "QUAD8": [0, 4, 1, 5, 2, 6, 3, 7]}


def deformed_outline(el):
    U = u[el.global_dofs()].reshape(-1, 2)
    return (el.X + SCALE * U)[OUTLINE[el.ref.name]]


s1 = np.array([el.principal_stress(u)[0] for el in elements]) / 1e6
s2 = np.array([el.principal_stress(u)[1] for el in elements]) / 1e6

div = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b52b2b"])
seq = LinearSegmentedColormap.from_list("seq", ["#0d366b", "#2a78d6", "#86b6ef", "#e9f1fc"])
lim = max(abs(s1.min()), abs(s1.max()))

fig, axes = plt.subplots(1, 2, figsize=(9, 5.4), sharey=True)
outline = np.array([[0, 0], [b, 0], [c, H], [0, H], [0, 0]])
panels = [
    (axes[0], s1, div, TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), r"$\sigma_1$ [MPa]", "(a) Tensión principal máxima"),
    (axes[1], s2, seq, Normalize(vmin=s2.min(), vmax=0.0), r"$\sigma_2$ [MPa]", "(b) Tensión principal mínima"),
]
for ax, vals, cmap, norm, label, title in panels:
    pc = PatchCollection([Polygon(deformed_outline(el), closed=True) for el in elements],
                         cmap=cmap, norm=norm, edgecolor="white", linewidth=0.3)
    pc.set_array(vals)
    ax.add_collection(pc)
    ax.plot(outline[:, 0], outline[:, 1], "--", color="#5f5e58", lw=1, label="Sin deformar")
    cb = fig.colorbar(pc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(label)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, b + 3)
    ax.set_ylim(-1.5, H + 2)
    ax.set_xlabel("x [m]")
    ax.set_title(title, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("y [m]")
axes[0].legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout()
plt.show()
