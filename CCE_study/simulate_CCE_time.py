import numpy as np
import math
import os
import argparse

from numba import njit
from scipy.special import erfc
from tqdm import tqdm


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Monte Carlo CCE simulation with Li trapping"
)

parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--N_charges", type=int, default=20)
parser.add_argument("--N_repeat", type=int, default=20)
parser.add_argument("--alpha", type=float, default=1.6e-9)
parser.add_argument("--FCCD", type=float, default=0.1)
parser.add_argument("--T_ann", type=float, default=623)
parser.add_argument("--t_ann", type=float, default=1080)
parser.add_argument("--sigma_R", type=float, default=1080)
parser.add_argument("--N_sat", type=float, default=1e14)
parser.add_argument("--output", type=str, default="library")

# Mantiene il comportamento del file allegato.
args = parser.parse_args([])


# ============================================================
# GEOMETRY [cm]
# ============================================================

Lx = 0.2
Ly = 0.2
Lz = 0.2

dx = 0.001
step = 0.005


# ============================================================
# DIFFUSION
# Valori e formula mantenuti come nel file allegato.
# ============================================================

D = 28.9
dt = 1.0
t_max = 10000

Nt = int(t_max / dt)
sigma = math.sqrt(2 * D * dt) * 1e-4


# ============================================================
# LITHIUM PARAMETERS
# ============================================================

r_Li = 0.012
cell_size = 2 * r_Li


# ============================================================
# ANNEALING MODEL
# ============================================================

R = 1.98
H = 11800
D0 = 2.5e-3


def update_parameters(args):

    params = {
        "T_ann": args.T_ann,
        "t_ann": args.t_ann,
        "alpha": args.alpha,
        "FCCD": args.FCCD,
        "sigma_R": args.sigma_R,
        "N_sat": args.N_sat,
    }

    params["Ns"] = 10**(
        21.27 - 2610 / params["T_ann"]
    )

    params["D_Li"] = D0 * math.exp(
        -H / (R * params["T_ann"])
    )

    return params


np.random.seed(args.seed)


# ============================================================
# LITHIUM PARTICLE OBJECT
# ============================================================

class LiParticle:

    __slots__ = ("x", "y", "z", "r")

    def __init__(self, x, y, z, r):
        self.x = x
        self.y = y
        self.z = z
        self.r = r


# ============================================================
# GRID INDEXING
# ============================================================

def get_index(x, y, z, nx, ny, nz):

    ix = int(x / cell_size)
    iy = int(y / cell_size)
    iz = int(z / cell_size)

    ix = max(0, min(ix, nx - 1))
    iy = max(0, min(iy, ny - 1))
    iz = max(0, min(iz, nz - 1))

    return ix, iy, iz


# ============================================================
# OVERLAP CHECK
# ============================================================

def is_overlapping(x, y, z, r, cells, nx, ny, nz):

    ix, iy, iz = get_index(x, y, z, nx, ny, nz)

    for dx_cell in (-1, 0, 1):
        for dy_cell in (-1, 0, 1):
            for dz_cell in (-1, 0, 1):

                jx = ix + dx_cell
                jy = iy + dy_cell
                jz = iz + dz_cell

                if (
                    jx < 0 or jy < 0 or jz < 0
                    or jx >= nx or jy >= ny or jz >= nz
                ):
                    continue

                for p in cells[jx][jy][jz]:

                    distance2 = (
                        (x - p.x)**2
                        + (y - p.y)**2
                        + (z - p.z)**2
                    )

                    if distance2 < (r + p.r)**2:
                        return True

    return False


def create_grid():

    nx = int(Lx / cell_size)
    ny = int(Ly / cell_size)
    nz = int(Lz / cell_size)

    cells = [
        [
            [[] for _ in range(nz)]
            for _ in range(ny)
        ]
        for _ in range(nx)
    ]

    return cells, nx, ny, nz


# ============================================================
# SATURATION DEPTH
# ============================================================

def calculate_saturation_depth(params):

    depth_list = np.arange(0, 0.11 + dx, dx)

    Nd_vals = params["Ns"] * erfc(
        depth_list
        / (2 * np.sqrt(params["D_Li"] * params["t_ann"]))
    )

    idx = np.where(Nd_vals <= params["N_sat"])[0]

    if len(idx) == 0:
        return depth_list[-1]

    return depth_list[idx[0]]


def truncated_poisson(lam, Nmax):

    if Nmax <= 0:
        return 0

    for _ in range(100):

        n = np.random.poisson(lam)

        if n <= Nmax:
            return n

    return Nmax


# ============================================================
# GENERATE LITHIUM
# ============================================================

def generate_lithium(params):

    saturation_depth = calculate_saturation_depth(params)
    cells, nx, ny, nz = create_grid()

    x_slices = np.arange(0, saturation_depth, dx)

    particles_x = []
    particles_y = []
    particles_z = []
    particles_R = []
    particles_slice = []

    total_particles = 0

    r0 = 20e-4
    V_particle = (4 / 3) * math.pi * r0**3

    radii_um = np.arange(20, 121, 1)
    radii = radii_um * 1e-4
    k = (radii / r0)**3

    P = np.zeros((len(radii), len(x_slices)))
    N20_per_x = np.zeros(len(x_slices), dtype=int)

    for j, xi in enumerate(
        tqdm(x_slices, desc="Calculating Li concentration")
    ):

        Nd_val = params["Ns"] * erfc(
            xi
            / (2 * np.sqrt(params["D_Li"] * params["t_ann"]))
        )

        density = params["alpha"] * max(
            Nd_val - params["N_sat"], 0
        )

        V_slice = dx * Ly * Lz
        lam = density * V_slice

        pf = 0.64
        Nmax = int(pf * V_slice / V_particle)

        N20_per_x[j] = int(truncated_poisson(lam, Nmax))

    # Stessa formula della distribuzione dei raggi del file.
    sigma_R = params["sigma_R"]

    print(sigma_R, "um")

    for j, xi in enumerate(x_slices):

        N20 = N20_per_x[j]

        if N20 <= 0:
            continue

        n_agg = N20 / k

        N_min_agg = 5
        possible = n_agg >= N_min_agg

        if not any(possible):

            R_max = r0
            i0 = np.argmin(np.abs(radii - r0))

            weights = np.zeros(len(radii))
            weights[i0] = 1.0

        else:

            R_max = r0 * (N20 / N_min_agg)**(1 / 3)

            preference = np.exp(
                -0.5 * ((radii - R_max) / sigma_R)**2
            )

            weights = np.zeros(len(radii))
            weights[possible] = preference[possible]

        total = weights.sum()

        if total > 0:
            P[:, j] = weights / total

    def get_radius(x_value):

        j = np.argmin(np.abs(x_slices - x_value))
        p = P[:, j]

        if p.sum() <= 0:
            return np.nan

        return np.random.choice(radii, p=p)

    N20_theoretical = np.zeros(len(x_slices), dtype=int)

    R_sampled_per_x = np.full(len(x_slices), np.nan)

    N_agglomerates_theoretical = np.zeros(
        len(x_slices), dtype=int
    )

    N_agglomerates_accepted = np.zeros(
        len(x_slices), dtype=int
    )

    N20_remaining_per_x = np.zeros(len(x_slices), dtype=int)
    N20_remaining_accepted = np.zeros(len(x_slices), dtype=int)

    for j, xi in enumerate(
        tqdm(x_slices, desc="Generating Li agglomerates")
    ):

        N20 = N20_per_x[j]
        N20_theoretical[j] = N20

        if N20 <= 0:
            continue

        R_sampled = get_radius(xi)

        if np.isnan(R_sampled):
            continue

        R_sampled_per_x[j] = R_sampled

        particles_per_agglomerate = (R_sampled / r0)**3

        N_R = int(np.floor(N20 / particles_per_agglomerate))
        N_agglomerates_theoretical[j] = N_R

        accepted = 0
        attempts = 0

        max_attempts = max(100, N_R * 200)

        while accepted < N_R and attempts < max_attempts:

            attempts += 1

            x_particle = xi + np.random.random() * dx
            y_particle = np.random.random() * Ly
            z_particle = np.random.random() * Lz

            if not is_overlapping(
                x_particle,
                y_particle,
                z_particle,
                R_sampled,
                cells,
                nx,
                ny,
                nz,
            ):

                ix, iy, iz = get_index(
                    x_particle, y_particle, z_particle,
                    nx, ny, nz,
                )

                p = LiParticle(
                    x_particle,
                    y_particle,
                    z_particle,
                    R_sampled,
                )

                cells[ix][iy][iz].append(p)

                particles_x.append(x_particle)
                particles_y.append(y_particle)
                particles_z.append(z_particle)
                particles_R.append(R_sampled)
                particles_slice.append(j)

                accepted += 1
                total_particles += 1

        N_agglomerates_accepted[j] = accepted

        # Logica del residuo mantenuta come nel file allegato.
        N20_used = N_R * particles_per_agglomerate

        N20_remaining = int(np.floor(N20 - N20_used))
        N20_remaining = max(N20_remaining, 0)

        N20_remaining_per_x[j] = N20_remaining

        accepted_primary = 0
        attempts_primary = 0

        max_attempts_primary = max(100, N20_remaining * 200)

        while (
            accepted_primary < N20_remaining
            and attempts_primary < max_attempts_primary
        ):

            attempts_primary += 1

            x_particle = xi + np.random.random() * dx
            y_particle = np.random.random() * Ly
            z_particle = np.random.random() * Lz

            R_primary = r0

            if not is_overlapping(
                x_particle,
                y_particle,
                z_particle,
                R_primary,
                cells,
                nx,
                ny,
                nz,
            ):

                ix, iy, iz = get_index(
                    x_particle, y_particle, z_particle,
                    nx, ny, nz,
                )

                p = LiParticle(
                    x_particle,
                    y_particle,
                    z_particle,
                    R_primary,
                )

                cells[ix][iy][iz].append(p)

                particles_x.append(x_particle)
                particles_y.append(y_particle)
                particles_z.append(z_particle)
                particles_R.append(R_primary)
                particles_slice.append(j)

                accepted_primary += 1
                total_particles += 1

        N20_remaining_accepted[j] = accepted_primary

    particles_x = np.array(particles_x)
    particles_y = np.array(particles_y)
    particles_z = np.array(particles_z)
    particles_R = np.array(particles_R)

    lithium_distribution = {
        "x": x_slices,
        "N20_theoretical": N20_theoretical,
        "R_sampled": R_sampled_per_x,
        "N_agglomerates_theoretical": N_agglomerates_theoretical,
        "N_agglomerates_accepted": N_agglomerates_accepted,
        "N20_remaining": N20_remaining_per_x,
        "N20_remaining_accepted": N20_remaining_accepted,
    }

    print(
        "Total theoretical primary Li (R=20 um) =",
        np.sum(N20_theoretical),
    )
    print(
        "Total theoretical agglomerates =",
        np.sum(N_agglomerates_theoretical),
    )
    print(
        "Total theoretical remaining 20 um particles =",
        np.sum(N20_remaining_per_x),
    )
    print(
        "Total accepted agglomerates =",
        np.sum(N_agglomerates_accepted),
    )
    print(
        "Total accepted remaining 20 um particles =",
        np.sum(N20_remaining_accepted),
    )
    print("Total accepted Li particles =", total_particles)

    return (
        cells,
        saturation_depth,
        particles_x,
        particles_y,
        particles_z,
        particles_R,
        lithium_distribution,
    )


# ============================================================
# NUMBA TRAPPING
# ============================================================

@njit
def check_trapping_numba(
    x,
    y,
    z,
    particle_x,
    particle_y,
    particle_z,
    particle_R,
):

    for i in range(len(particle_x)):

        distance2 = (
            (x - particle_x[i])**2
            + (y - particle_y[i])**2
            + (z - particle_z[i])**2
        )

        if distance2 < particle_R[i]**2:
            return True

    return False


# ============================================================
# CHARGE TRANSPORT
#
# status:
# 0 = still moving at t_max
# 1 = collected at FCCD
# 2 = lost in dead layer
# 3 = lost on lithium
# ============================================================

@njit
def charge_transport_numba(
    x0,
    N,
    particle_x,
    particle_y,
    particle_z,
    particle_R,
    dx,
    Ly,
    Lz,
    Nt,
    sigma,
    FCCD,
    dt,
    seed,
):

    np.random.seed(seed)

    elapsed_ns = np.full(N, Nt * dt, dtype=np.float64)
    status = np.zeros(N, dtype=np.int8)

    for n in range(N):

        x = x0
        y = np.random.random() * Ly
        z = np.random.random() * Lz

        for k in range(Nt + 1):

            if x >= FCCD:
                elapsed_ns[n] = k * dt
                status[n] = 1
                break

            if x <= dx:
                elapsed_ns[n] = k * dt
                status[n] = 2
                break

            if check_trapping_numba(
                x,
                y,
                z,
                particle_x,
                particle_y,
                particle_z,
                particle_R,
            ):
                elapsed_ns[n] = k * dt
                status[n] = 3
                break

            if k == Nt:
                break

            x += sigma * np.random.randn()
            y += sigma * np.random.randn()
            z += sigma * np.random.randn()

            y = y % (2 * Ly)
            if y > Ly:
                y = 2 * Ly - y

            z = z % (2 * Lz)
            if z > Lz:
                z = 2 * Lz - z

    return elapsed_ns, status


# ============================================================
# CCE SIMULATION
# ============================================================

def simulate_CCE(
    cells,
    saturation_depth,
    particle_x,
    particle_y,
    particle_z,
    particle_R,
    args,
):

    print("\nStarting CCE and timing simulation...\n")

    arrays = tuple(
        np.asarray(a, dtype=np.float64)
        for a in (
            particle_x,
            particle_y,
            particle_z,
            particle_R,
        )
    )

    if any(a.ndim != 1 for a in arrays):
        raise ValueError("Coordinates and radii must be 1D arrays.")

    if not all(len(a) == len(arrays[0]) for a in arrays):
        raise ValueError("Coordinates and radii must have equal lengths.")

    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Coordinates and radii must be finite.")

    if np.any(arrays[3] <= 0):
        raise ValueError("Particle radii must be positive.")

    if args.N_charges <= 0 or args.N_repeat <= 0:
        raise ValueError("N_charges and N_repeat must be positive.")

    if (
        not np.isfinite(dt) or dt <= 0
        or Nt < 0
        or not np.isfinite(sigma) or sigma < 0
        or Ly <= 0 or Lz <= 0
        or args.FCCD <= dx
    ):
        raise ValueError("Invalid transport parameters.")

    (
        particle_x,
        particle_y,
        particle_z,
        particle_R,
    ) = tuple(np.ascontiguousarray(a) for a in arrays)

    x_positions = np.arange(
        0,
        args.FCCD + 3 * step,
        step,
    )

    CCE_matrix = np.zeros(
        (len(x_positions), args.N_repeat)
    )

    # Axes: [depth, repetition, carrier]
    shape = (
        len(x_positions),
        args.N_repeat,
        args.N_charges,
    )

    elapsed_ns = np.empty(shape, dtype=np.float64)
    status = np.empty(shape, dtype=np.int8)

    rng = np.random.default_rng(int(args.seed))

    transport_seeds = rng.integers(
        0,
        2**31 - 1,
        size=(len(x_positions), args.N_repeat),
        dtype=np.int64,
    )

    for i, x0 in enumerate(
        tqdm(x_positions, desc="CCE depth")
    ):
        for j in range(args.N_repeat):

            times, outcomes = charge_transport_numba(
                x0,
                args.N_charges,
                particle_x,
                particle_y,
                particle_z,
                particle_R,
                dx,
                Ly,
                Lz,
                Nt,
                sigma,
                args.FCCD,
                dt,
                int(transport_seeds[i, j]),
            )

            elapsed_ns[i, j, :] = times
            status[i, j, :] = outcomes

            CCE_matrix[i, j] = (
                np.count_nonzero(outcomes == 1)
                / args.N_charges
            )

    CCE_mean = np.mean(CCE_matrix, axis=1)
    CCE_std = np.std(CCE_matrix, axis=1)

    print("\nTrajectory outcomes:")

    labels = (
        "Still moving at t_max",
        "Collected at FCCD",
        "Lost in dead layer",
        "Lost on lithium",
    )

    for code, label in enumerate(labels):
        print(f"{label}: {np.count_nonzero(status == code)}")

    return (
        x_positions,
        CCE_matrix,
        CCE_mean,
        CCE_std,
        elapsed_ns,
        status,
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    x_positions,
    CCE_matrix,
    CCE_mean,
    CCE_std,
    saturation_depth,
    particle_x,
    particle_y,
    particle_z,
    particle_R,
    elapsed_ns,
    status,
    args,
    params,
):

    os.makedirs(args.output, exist_ok=True)

    filename = os.path.join(
        args.output,
        f"curve_{int(args.seed):06d}.npz",
    )

    np.savez_compressed(
        filename,

        # CCE
        x=x_positions,
        CCE=CCE_matrix,
        CCE_mean=CCE_mean,
        CCE_std=CCE_std,

        # Times and outcomes: [depth, repetition, carrier]
        elapsed_ns=elapsed_ns,
        status=status,
        time_array_axes=np.array(
            ["depth", "repetition", "carrier"]
        ),
        status_codes=np.array([0, 1, 2, 3], dtype=np.int8),
        status_labels=np.array([
            "still_moving",
            "collected_FCCD",
            "lost_dead_layer",
            "lost_lithium",
        ]),
        t0_ns=0.0,
        t_max_ns=Nt * dt,

        # Lithium [cm]
        particle_x=particle_x,
        particle_y=particle_y,
        particle_z=particle_z,
        particle_R=particle_R,

        # Detector [cm]
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        dx_cm=dx,
        FCCD_cm=args.FCCD,

        # Lithium parameters
        alpha=params["alpha"],
        T_ann=params["T_ann"],
        t_ann=params["t_ann"],
        Ns=params["Ns"],
        D_Li=params["D_Li"],
        Nd_saturation=params["N_sat"],
        saturation_depth=saturation_depth,
        sigma_R=params["sigma_R"],

        # Simulation parameters
        N_charges=args.N_charges,
        N_repeat=args.N_repeat,
        seed=int(args.seed),
        D=D,
        dt=dt,
        Nt=Nt,
        sigma=sigma,
    )

    print(f"\nSaved: {filename}\n")

    return filename


# ============================================================
# MAIN
# ============================================================

def main():

    params = update_parameters(args)
    np.random.seed(int(args.seed))

    (
        cells,
        saturation_depth,
        particle_x,
        particle_y,
        particle_z,
        particle_R,
        lithium_distribution,
    ) = generate_lithium(params)

    (
        x_positions,
        CCE_matrix,
        CCE_mean,
        CCE_std,
        elapsed_ns,
        status,
    ) = simulate_CCE(
        cells,
        saturation_depth,
        particle_x,
        particle_y,
        particle_z,
        particle_R,
        args,
    )

    save_results(
        x_positions,
        CCE_matrix,
        CCE_mean,
        CCE_std,
        saturation_depth,
        particle_x,
        particle_y,
        particle_z,
        particle_R,
        elapsed_ns,
        status,
        args,
        params,
    )


if __name__ == "__main__":
    main()