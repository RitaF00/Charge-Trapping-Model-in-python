import cce_library as cce
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--N_charges", type=int, required=True)
args = parser.parse_args()

N_charges = args.N_charges


library, history = cce.build_library(
    fccd_mm=0.95,
    alpha=1e-10,
    dx_um=1,
    ncharges=N_charges,
    save_history=True
)

np.savez(
    "CCE_library_fast.npz",

    depth_mm=library.depth_mm,

    cce=library.cce,

    response=library.response,

    fccd_mm=library.fccd_mm,

    alpha=library.alpha,

    dx_um=library.dx_um
)


# ==========================
# Salva history
# ==========================

np.savez(
    "Charge_history_fast.npz",

    depth_mm=history.depth_mm,

    lifetime_ns=history.lifetime_ns,

    collected=history.collected,

    trapped=history.trapped
)


print("Salvataggio completato")




depth = library.depth_mm
CCE = library.cce




plt.figure(figsize=(7,5))

plt.plot(
    library.depth_mm,
    library.cce,
    lw=2
)

plt.xlabel("Depth [mm]")
plt.ylabel("CCE")
plt.grid()
plt.savefig("plot/all_CCE.png", dpi = 300)


cce_mean = np.mean(
    library.cce,
    axis=1
)

plt.figure(figsize=(7,5))

plt.plot(
    library.depth_mm,
    cce_mean,
    lw=1,
    color = 'red',
    label="Mean CCE",
    zorder = 10
)


for r in range(library.cce.shape[1]):

    plt.plot(
        library.depth_mm,
        library.cce[:,r],
        alpha=0.4
    )


plt.xlabel("Depth [mm]")
plt.ylabel("CCE")
plt.grid()
plt.legend()
plt.savefig("plot/all_CCE_and_mean.png", dpi = 300)



# forza stile bianco
plt.style.use("default")


# ==========================
# Prepare data
# ==========================

depth = np.repeat(
    library.depth_mm,
    library.cce.shape[1]
)

cce_values = library.cce.flatten()



# ==========================
# Plot
# ==========================

fig, ax = plt.subplots(
    figsize=(8,6),
    facecolor="white"
)


ax.set_facecolor("white")


cmap = plt.cm.viridis.copy()
cmap.set_under("white")


h = ax.hist2d(
    depth,
    cce_values,
    bins=(50,N_charges),
    cmap=cmap,
    vmin=1
)

h[3].set_clim(1, None)

# colorbar

cbar = fig.colorbar(
    h[3],
    ax=ax
)

cbar.set_label(
    "Counts",
    fontsize=13
)



# ==========================
# Mean CCE
# ==========================

cce_mean = np.mean(
    library.cce,
    axis=1
)


ax.plot(
    library.depth_mm,
    cce_mean,
    color="red",
    lw=1,
    label="Mean CCE"
)



# ==========================
# Labels
# ==========================

ax.set_xlabel(
    "Depth [mm]",
    fontsize=14
)

ax.set_ylabel(
    "CCE",
    fontsize=14
)


ax.set_title(
    "CCE distribution vs depth",
    fontsize=15
)



# ==========================
# Style
# ==========================

ax.tick_params(
    labelsize=12
)


ax.legend(
    fontsize=12,
    frameon=True
)


for spine in ax.spines.values():
    spine.set_linewidth(1.2)



plt.tight_layout()
plt.savefig("plot/CCE_distr.png", dpi = 300)
