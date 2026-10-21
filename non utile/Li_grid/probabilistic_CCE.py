import numpy as np
import math
import json
from scipy.special import erfc
from scipy.stats import poisson
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from numba import njit
import argparse




parser = argparse.ArgumentParser()
parser.add_argument("--N_charges", type=int, required=True)
args = parser.parse_args()

N_charges = args.N_charges
#N_repeat = args.N_repeat

# ============================================================
# GEOMETRIC PARAMETERS (cm)
# ============================================================
Lx = 0.2
Ly = 0.2
Lz = 0.2

dx = 0.001
FCCD_cm = 0.1

# ============================================================
# DIFFUSION
# ============================================================
D = 28.9
dt = 1.0
t_max = 10000
Nt = int(t_max / dt)
sigma = math.sqrt(6 * D * dt) * 1e-4

# ============================================================
# LITHIUM
# ============================================================
r_Li = 0.002   # cm
cell_size = r_Li


# ============================================================
# ANNEALING
# ============================================================

'''
valori per detector ORTEC e non MIRION
t_ann = 18 * 60
T_ann = 623
'''

t_ann = 2 * 60 * 60
T_ann = 473.15

R = 1.98
H = 11800
D0 = 2.5e-3

Ns = 10 ** (21.27 - 2610 / T_ann)
D_Li = D0 * math.exp(-H / (R * T_ann))

Nd_saturation = 1e14
alpha = 5e-10


# ============================================================
# SATURATION DEPTH
# ============================================================
depth_list = np.arange(0, 0.11 + dx, dx)
Nd_vals = Ns * erfc(depth_list / (2 * np.sqrt(D_Li * t_ann)))

idx = np.where(Nd_vals - Nd_saturation <= 0)[0][0]
saturation_depth = depth_list[idx]

print("Saturation depth =", saturation_depth * 10, "mm")

# ============================================================
# LITHIUM PARTICLE
# ============================================================
class LiParticle:
    __slots__ = ("x", "y", "z", "r")

    def __init__(self, x, y, z, r):
        self.x = x
        self.y = y
        self.z = z
        self.r = r


# ============================================================
# TRUNCATED POISSON
# ============================================================
def truncated_poisson(lam, Nmax):
    if Nmax <= 0:
        return 0

    for _ in range(100):
        n = np.random.poisson(lam)
        if n <= Nmax:
            return n

    return Nmax



# ============================================================
# GRID INDEXING
# ============================================================
def get_index(x, y, z, nx, ny, nz):
    ix = min(max(int(x / cell_size), 0), nx - 1)
    iy = min(max(int(y / cell_size), 0), ny - 1)
    iz = min(max(int(z / cell_size), 0), nz - 1)
    return ix, iy, iz



# ============================================================
# OVERLAP CHECK
# ============================================================
def is_overlapping(x, y, z, r, cells, nx, ny, nz):
    ix, iy, iz = get_index(x, y, z, nx, ny, nz)

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                jx, jy, jz = ix + dx, iy + dy, iz + dz

                if 0 <= jx < nx and 0 <= jy < ny and 0 <= jz < nz:
                    for p in cells[jx][jy][jz]:
                        if (x - p.x)**2 + (y - p.y)**2 + (z - p.z)**2 < (r + p.r)**2:
                            return True
    return False




# ============================================================
# LITHIUM GENERATION
# ============================================================
def generate_lithium():
    nx = int(Lx / cell_size)
    ny = int(Ly / cell_size)
    nz = int(Lz / cell_size)

    cells = [[[[] for _ in range(nz)] for _ in range(ny)] for _ in range(nx)]

    x_slices = np.arange(0, Lx + dx, dx)
    total = 0

    for xi in x_slices:
        if xi >= saturation_depth:
            continue

        Nd_val = Ns * erfc(xi / (2 * np.sqrt(D_Li * t_ann)))
        density = alpha * max(Nd_val - Nd_saturation, 0.0)

        lam = density * dx * Ly * Lz

        V_particle = (4/3) * math.pi * r_Li**3
        V_slice = dx * Ly * Lz
        N_max = int(V_slice / V_particle)

        N_i = truncated_poisson(lam, N_max)

        accepted = 0
        trials = N_i * 100

        for _ in range(trials):
            if accepted >= N_i:
                break

            x = xi + np.random.rand() * dx
            y = np.random.rand() * Ly
            z = np.random.rand() * Lz

            if not is_overlapping(x, y, z, r_Li, cells, nx, ny, nz):
                ix, iy, iz = get_index(x, y, z, nx, ny, nz)
                cells[ix][iy][iz].append(LiParticle(x, y, z, r_Li))
                accepted += 1
                total += 1

    print("Total Li particles =", total)
    return cells



# ============================================================
# GENERATE LITHIUM
# ============================================================
cells = generate_lithium()

# ============================================================
# CHARGE TRANSPORT + TRAPPING
# ============================================================
def charge_transport_3D(x_charges, N, cells,
                        x_saturation, dx,
                        Lx, Ly, Lz,
                        Nt, sigma, FCCD_cm, r):

    if np.isscalar(x_charges):
        x_charges = np.full(N, x_charges)

    nx = len(cells)
    ny = len(cells[0])
    nz = len(cells[0][0])

    collected = 0
    trapped = 0

    for n in range(N):
        x = x_charges[n]
        y = np.random.rand() * Ly
        z = np.random.rand() * Lz

        is_trapped = False

        for _ in range(Nt):
            x += sigma * np.random.randn()
            y += sigma * np.random.randn()
            z += sigma * np.random.randn()

            x = np.clip(x, 0, Lx)
            y = np.clip(y, 0, Ly)
            z = np.clip(z, 0, Lz)

            if x >= FCCD_cm:
                collected += 1
                break

            if x <= dx:
                is_trapped = True
                break

            if x <= x_saturation:
                ix, iy, iz = get_index(x, y, z, nx, ny, nz)

                for dx_ in (-1, 0, 1):
                    for dy_ in (-1, 0, 1):
                        for dz_ in (-1, 0, 1):
                            jx, jy, jz = ix + dx_, iy + dy_, iz + dz_

                            if 0 <= jx < nx and 0 <= jy < ny and 0 <= jz < nz:
                                for p in cells[jx][jy][jz]:
                                    if (x - p.x)**2 + (y - p.y)**2 + (z - p.z)**2 < r**2:
                                        is_trapped = True
                                        break

                            if is_trapped:
                                break
                        if is_trapped:
                            break

            if is_trapped:
                trapped += 1
                break

    return collected, trapped


# ============================================================
# CCE SIMULATION
# ============================================================
# ============================================================
# CCE SIMULATION
# ============================================================
dx = 0.001
x_pos = np.arange(0, 0.11 + dx, dx)
#N_charges = 250
N_repeat = 50



CCE_matrix = np.zeros((len(x_pos), N_repeat))

for i, x0 in enumerate(tqdm(x_pos)):
    for j in range(N_repeat):
        collected, _ = charge_transport_3D(
            x0, N_charges, cells,
            saturation_depth, dx,
            Lx, Ly, Lz,
            Nt, sigma, FCCD_cm, r_Li
        )
        CCE_matrix[i, j] = collected / N_charges


# ============================================================
# STATISTICS
# ============================================================
CCE_mean = CCE_matrix.mean(axis=1)
CCE_std = CCE_matrix.std(axis=1)

# plot mean
fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

# Main curve
ax.plot(
    x_pos * 10000,
    CCE_mean,
    lw=2.2,
    color="black",
    label="CCE"
)

# Uncertainty band (more transparent and cleaner)
ax.fill_between(
    x_pos * 10000,
    CCE_mean - CCE_std,
    CCE_mean + CCE_std,
    alpha=0.52,
    color="gray",
    linewidth=0
)

# Labels (slightly larger + clearer units)
ax.set_xlabel("Depth (um)", fontsize=11)
ax.set_ylabel("Charge Collection Efficiency (CCE)", fontsize=11)

# Grid (major only, subtle)
ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)

# Improve axes limits (removes useless whitespace)
ax.set_xlim(x_pos.min() * 10000, x_pos.max() * 10000)
ax.set_ylim(-0.05, 1.05)

ax.axvline(0.263 * 1000, 0, 1, ls = '--', color = 'deepskyblue' )
# Tick styling
ax.tick_params(axis="both", which="major", labelsize=10, width=1)

ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("plot/mean_std_CCE.png", dpi = 300)


# 3D


# Bin della CCE
n_cce_bins = N_charges 
cce_bins = np.linspace(0, 1, n_cce_bins + 1)

# Matrice delle frequenze
density = np.zeros((n_cce_bins, len(x_pos)))

for i in range(len(x_pos)):
    density[:, i], _ = np.histogram(CCE_matrix[i, :], bins=cce_bins)

# Plot
fig, ax = plt.subplots(figsize=(5, 5), dpi=150)


dx = x_pos[1] - x_pos[0]

x_edges = np.concatenate([
    [x_pos[0] - dx/2],
    x_pos[:-1] + dx/2,
    [x_pos[-1] + dx/2]
])

plt.pcolormesh(
    x_edges*10,
    cce_bins,
    density,
    cmap='viridis',
    norm=LogNorm(vmin=1, vmax=max(2, density.max())),
    shading='flat'
)

ax.plot(
    x_pos * 10,
    CCE_mean,
    lw=2.2,
    color="red",
    label="CCE"
)

plt.xlabel("Depth [cm]")
plt.ylabel("CCE")
plt.colorbar(label="Counts")

plt.tight_layout()
plt.savefig("plot/distrib_CCE.png", dpi = 300)



# sovrapposizi<one

fig, ax = plt.subplots(figsize=(6,5), dpi=150)

# tutte le realizzazioni Monte Carlo
for j in range(N_repeat):
    ax.plot(
        x_pos*10,
        CCE_matrix[:, j],
        color="black",
        alpha=0.15,
        lw=1
    )

# media
CCE_mean = np.mean(CCE_matrix, axis=1)

ax.plot(
    x_pos*10,
    CCE_mean,
    color="orange",
    lw=2.5,
    label="Mean CCE"
)

# deviazione standard
CCE_std = np.std(CCE_matrix, axis=1)

ax.fill_between(
    x_pos*10,
    CCE_mean-CCE_std,
    CCE_mean+CCE_std,
    alpha=0.25,
    label="±1σ"
)

ax.set_xlabel("Depth [cm]")
ax.set_ylabel("CCE")
ax.set_ylim(0,1.05)

ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("plot/sovrapp_CCE.png")


np.savez(
    "plot/CCE_data.npz",
    x_pos=x_pos,
    CCE_matrix=CCE_matrix,
    CCE_mean=CCE_mean,
    CCE_std=CCE_std
)

print("Dati CCE salvati in plot/CCE_data.npz)


