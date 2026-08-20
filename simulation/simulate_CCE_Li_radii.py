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


parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="Random seed"
)

parser.add_argument(
    "--N_charges",
    type=int,
    default=20,
    help="Number of charge carriers"
)

parser.add_argument(
    "--N_repeat",
    type=int,
    default=20,
    help="Number of repetitions per depth"
)

parser.add_argument(
    "--alpha",
    type=float,
    default=1.6e-9,
    help="Li precipitation coefficient"
)

parser.add_argument(
    "--FCCD",
    type=float,
    default=0.1,
    help="Full charge collection depth"
)

parser.add_argument(
    "--T_ann",
    type=float,
    default=623,
    help="Annealing temperature [K]"
)

parser.add_argument(
    "--t_ann",
    type=float,
    default=1080,
    help="Annealing time [s]"
)

parser.add_argument(
    "--output",
    type=str,
    default="library",
    help="Output directory"
)


# Used only when running the .py directly
args = parser.parse_args([])



# ============================================================
# GEOMETRY (cm)
# ============================================================

Lx = 0.2
Ly = 0.2
Lz = 0.2


dx = 0.001      # cm = 10 um

step = 0.005    # cm






# ============================================================
# DIFFUSION
# ============================================================

D = 28.9       # cm2/ns

dt = 1.0       # ns

t_max = 10000

Nt = int(t_max/dt)


sigma = math.sqrt(2*D*dt)*1e-4   # evoluzione indipendente lungo le 3 direzioni



# ============================================================
# LITHIUM PARAMETERS
# ============================================================

r_Li = 0.002       # cm
r_Li =0.012        # MAXIMUM RADII 
cell_size = 2 * r_Li



# ============================================================
# ANNEALING MODEL
# ============================================================

R = 1.98

H = 11800

D0 = 2.5e-3


Nd_saturation = 1e14



# ============================================================
# PARAMETER UPDATE
# ============================================================

def update_parameters(args):

    """
    Compute parameters depending on annealing conditions
    """

    params = {}

    params["T_ann"] = args.T_ann

    params["t_ann"] = args.t_ann

    params["alpha"] = args.alpha

    params["FCCD"] = args.FCCD

    params["Ns"] = 10**(
        21.27 - 2610/params["T_ann"]
    )


    params["D_Li"] = D0 * math.exp(
        -H/(R*params["T_ann"])
    )


    return params



# ============================================================
# RANDOM INFO
# ============================================================

np.random.seed(args.seed)


print("--------------------------------")
print("CCE simulation")
print("--------------------------------")
print("seed =", args.seed)
print("--------------------------------")



# ============================================================
# LITHIUM PARTICLE OBJECT
# ============================================================

class LiParticle:

    __slots__ = (
        "x",
        "y",
        "z",
        "r"
    )


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



    ix = max(0, min(ix, nx-1))

    iy = max(0, min(iy, ny-1))

    iz = max(0, min(iz, nz-1))


    return ix, iy, iz



# ============================================================
# OVERLAP CHECK
# ============================================================

def is_overlapping(
        x,
        y,
        z,
        r,
        cells,
        nx,
        ny,
        nz
):


    ix, iy, iz = get_index(
        x,
        y,
        z,
        nx,
        ny,
        nz
    )



    for dx_cell in (-1,0,1):

        for dy_cell in (-1,0,1):

            for dz_cell in (-1,0,1):


                jx = ix + dx_cell

                jy = iy + dy_cell

                jz = iz + dz_cell



                if (
                    jx < 0 or
                    jy < 0 or
                    jz < 0 or
                    jx >= nx or
                    jy >= ny or
                    jz >= nz
                ):
                    continue



                for p in cells[jx][jy][jz]:


                    distance2 = (
                        (x-p.x)**2 +
                        (y-p.y)**2 +
                        (z-p.z)**2
                    )


                    if distance2 < (r+p.r)**2:

                        return True



    return False



# ============================================================
# GRID CREATION
# ============================================================

def create_grid():


    nx = int(Lx/cell_size)

    ny = int(Ly/cell_size)

    nz = int(Lz/cell_size)



    cells = [
        [
            [
                []
                for _ in range(nz)
            ]
            for _ in range(ny)
        ]
        for _ in range(nx)
    ]


    return cells, nx, ny, nz










# ============================================================
# SATURATION DEPTH
# ============================================================

def calculate_saturation_depth(params):


    depth_list = np.arange(
        0,
        0.11 + dx,
        dx
    )


    Nd_vals = params["Ns"] * erfc(
        depth_list /
        (2*np.sqrt(params["D_Li"] * params["t_ann"]))
    )


    idx = np.where(
        Nd_vals <= Nd_saturation
    )[0]


    if len(idx) == 0:

        saturation_depth = depth_list[-1]

    else:

        saturation_depth = depth_list[idx[0]]

    #print(f"saturation depth {saturation_depth} cm")


    return saturation_depth




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
# SATURATION DEPTH
# ============================================================

def calculate_saturation_depth(params):

    depth_list = np.arange(
        0,
        0.11 + dx,
        dx
    )

    Nd_vals = (
        params["Ns"] *
        erfc(
            depth_list /
            (
                2 *
                np.sqrt(
                    params["D_Li"] *
                    params["t_ann"]
                )
            )
        )
    )

    idx = np.where(
        Nd_vals <= Nd_saturation
    )[0]

    if len(idx) == 0:

        saturation_depth = depth_list[-1]

    else:

        saturation_depth = depth_list[idx[0]]

    return saturation_depth


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
# GENERATE LITHIUM
# ============================================================

def generate_lithium(params):

    print("")
    print("Generating Li distribution...")

    # ========================================================
    # SATURATION DEPTH
    # ========================================================

    saturation_depth = calculate_saturation_depth(params)

    print(
        "Saturation depth =",
        saturation_depth * 10,
        "mm"
    )

    # ========================================================
    # GRID
    # ========================================================

    cells, nx, ny, nz = create_grid()

    # ========================================================
    # X SLICES
    # ========================================================

    x_slices = np.arange(
        0,
        saturation_depth,
        dx
    )

    # ========================================================
    # OUTPUT COORDINATES
    # ========================================================

    particles_x = []
    particles_y = []
    particles_z = []

    total_particles = 0

    # ========================================================
    # PRIMARY Li RADIUS
    #
    # 20 um = 20e-4 cm = 2e-3 cm
    # ========================================================

    r0 = 20e-4       # cm
    # equivalent to:
    # r0 = 2e-3 cm

    # ========================================================
    # VOLUME OF ONE PRIMARY 20 um PARTICLE
    # ========================================================

    V_particle = (
        4 / 3 *
        math.pi *
        r0**3
    )

    # ========================================================
    # POSSIBLE AGGLOMERATE RADII
    #
    # 20 -> 120 um
    #
    # converted to cm
    # ========================================================

    radii_um = np.arange(
        20,
        121,
        5
    )

    radii = (
        radii_um *
        1e-4
    )

    # ========================================================
    # NUMBER OF PRIMARY 20 um PARTICLES REQUIRED
    # FOR EACH AGGLOMERATE RADIUS
    #
    # k = (R/r0)^3
    # ========================================================

    k = (
        radii /
        r0
    )**3

    # ========================================================
    # TARGET NUMBER OF AGGLOMERATES
    # ========================================================

    N_target_agg = 4

    # ========================================================
    # PROBABILITY MATRIX
    #
    # P[radius, x]
    # ========================================================

    P = np.zeros(
        (
            len(radii),
            len(x_slices)
        )
    )

    # ========================================================
    # 1. THEORETICAL NUMBER OF 20 um Li
    # ========================================================

    N20_per_x = np.zeros(
        len(x_slices),
        dtype=int
    )

    print("")

    for j, xi in enumerate(
        tqdm(
            x_slices,
            desc="Calculating Li concentration"
        )
    ):

        # ----------------------------------------------------
        # Li concentration profile
        # ----------------------------------------------------

        Nd_val = (
            params["Ns"] *
            erfc(
                xi /
                (
                    2 *
                    np.sqrt(
                        params["D_Li"] *
                        params["t_ann"]
                    )
                )
            )
        )

        # ----------------------------------------------------
        # Particle density
        # ----------------------------------------------------

        density = (
            params["alpha"] *
            max(
                Nd_val -
                Nd_saturation,
                0
            )
        )

        # ----------------------------------------------------
        # Slice volume
        # ----------------------------------------------------

        V_slice = (
            dx *
            Ly *
            Lz
        )

        # ----------------------------------------------------
        # Expected number of PRIMARY 20 um particles
        # ----------------------------------------------------

        lam = (
            density *
            V_slice
        )

        # ----------------------------------------------------
        # Maximum geometrical packing
        # ----------------------------------------------------

        pf = 0.64

        Nmax = int(
            pf *
            V_slice /
            V_particle
        )

        # ----------------------------------------------------
        # Theoretical number of 20 um particles
        # ----------------------------------------------------

        N20_per_x[j] = int(
            truncated_poisson(
                lam,
                Nmax
            )
        )

    # ========================================================
    # 2. CALCULATE P(R | x)
    # ========================================================

    sigma = np.random.uniform(
        20e-4,
        60e-4
    )
    # 20-60 um expressed in cm

    for j, xi in enumerate(x_slices):

        N20 = N20_per_x[j]

        if N20 <= 0:
            continue

        # ----------------------------------------------------
        # Number of agglomerates possible for each radius
        # ----------------------------------------------------

        n_agg = (
            N20 /
            k
        )

        # ----------------------------------------------------
        # Physically possible radii
        # ----------------------------------------------------

        possible = (
            n_agg >= 1
        )

        # ----------------------------------------------------
        # Target number of agglomerates
        # ----------------------------------------------------

        N_agg_target = min(
            N20,
            N_target_agg
        )

        # ----------------------------------------------------
        # Radius conserving total volume
        # ----------------------------------------------------

        R_target = (
            r0 *
            (
                N20 /
                N_agg_target
            )**(1 / 3)
        )

        # ----------------------------------------------------
        # Preference for target radius
        # ----------------------------------------------------

        preference = np.exp(
            -0.5 *
            (
                (
                    radii -
                    R_target
                ) /
                sigma
            )**2
        )

        # ----------------------------------------------------
        # Weights
        # ----------------------------------------------------

        weights = np.zeros(
            len(radii)
        )

        weights[possible] = (
            preference[possible]
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        total = weights.sum()

        if total > 0:

            P[:, j] = (
                weights /
                total
            )

    # ========================================================
    # 3. FUNCTION: SAMPLE RADIUS AT x
    # ========================================================

    def get_radius(x_value):

        # ----------------------------------------------------
        # Find closest x slice
        # ----------------------------------------------------

        j = np.argmin(
            np.abs(
                x_slices -
                x_value
            )
        )

        # ----------------------------------------------------
        # Probability distribution
        # ----------------------------------------------------

        p = P[:, j]

        # ----------------------------------------------------
        # No valid distribution
        # ----------------------------------------------------

        if p.sum() <= 0:

            return np.nan

        # ----------------------------------------------------
        # Sample radius
        # ----------------------------------------------------

        return np.random.choice(
            radii,
            p=p
        )

    # ========================================================
    # 4. ARRAYS TO SAVE RESULTS
    # ========================================================

    # --------------------------------------------------------
    # Theoretical number of primary particles with r = 20 um
    # --------------------------------------------------------

    N20_theoretical = np.zeros(
        len(x_slices),
        dtype=int
    )

    # --------------------------------------------------------
    # Sampled agglomerate radius [cm]
    # --------------------------------------------------------

    R_sampled_per_x = np.full(
        len(x_slices),
        np.nan
    )

    # --------------------------------------------------------
    # Theoretical number of agglomerates
    # --------------------------------------------------------

    N_agglomerates_theoretical = np.zeros(
        len(x_slices),
        dtype=int
    )

    # --------------------------------------------------------
    # Actually generated agglomerates
    # --------------------------------------------------------

    N_agglomerates_accepted = np.zeros(
        len(x_slices),
        dtype=int
    )

    # ========================================================
    # 5. GENERATE AGGLOMERATES
    # ========================================================

    print("")

    for j, xi in enumerate(
        tqdm(
            x_slices,
            desc="Generating Li agglomerates"
        )
    ):

        # ----------------------------------------------------
        # Theoretical number of 20 um Li
        # ----------------------------------------------------

        N20 = N20_per_x[j]

        N20_theoretical[j] = N20

        if N20 <= 0:
            continue

        # ----------------------------------------------------
        # Sample agglomerate radius
        # ----------------------------------------------------

        R_sampled = get_radius(
            xi
        )

        if np.isnan(
            R_sampled
        ):
            continue

        # ----------------------------------------------------
        # SAVE RADIUS [cm]
        # ----------------------------------------------------

        R_sampled_per_x[j] = (
            R_sampled
        )

        # ----------------------------------------------------
        # VOLUME CONSERVATION
        #
        # N20 * V20 = NR * VR
        #
        # therefore:
        #
        # NR = N20 * (r0/R)^3
        # ----------------------------------------------------

        N_R = int(
            np.floor(
                N20 *
                (
                    r0 /
                    R_sampled
                )**3
            )
        )

        # ----------------------------------------------------
        # SAVE THEORETICAL NUMBER
        # ----------------------------------------------------

        N_agglomerates_theoretical[j] = (
            N_R
        )

        # ====================================================
        # GENERATE N_R AGGLOMERATES
        # ====================================================

        accepted = 0
        attempts = 0

        max_attempts = max(
            100,
            N_R * 200
        )

        while (
            accepted < N_R and
            attempts < max_attempts
        ):

            attempts += 1

            # ------------------------------------------------
            # Random position inside x slice
            # ------------------------------------------------

            x_particle = (
                xi +
                np.random.random() *
                dx
            )

            y_particle = (
                np.random.random() *
                Ly
            )

            z_particle = (
                np.random.random() *
                Lz
            )

            # ------------------------------------------------
            # OVERLAP CHECK
            #
            # R_sampled is in cm
            # ------------------------------------------------

            if not is_overlapping(
                x_particle,
                y_particle,
                z_particle,
                R_sampled,
                cells,
                nx,
                ny,
                nz
            ):

                # ------------------------------------------------
                # Grid index
                # ------------------------------------------------

                ix, iy, iz = get_index(
                    x_particle,
                    y_particle,
                    z_particle,
                    nx,
                    ny,
                    nz
                )

                # ------------------------------------------------
                # Create agglomerate
                # ------------------------------------------------

                p = LiParticle(
                    x_particle,
                    y_particle,
                    z_particle,
                    R_sampled
                )

                # ------------------------------------------------
                # Add to grid
                # ------------------------------------------------

                cells[ix][iy][iz].append(
                    p
                )

                # ------------------------------------------------
                # Save coordinates
                # ------------------------------------------------

                particles_x.append(
                    x_particle
                )

                particles_y.append(
                    y_particle
                )

                particles_z.append(
                    z_particle
                )

                accepted += 1

                total_particles += 1

        # ----------------------------------------------------
        # Save actually accepted agglomerates
        # ----------------------------------------------------

        N_agglomerates_accepted[j] = (
            accepted
        )

    # ========================================================
    # 6. CONVERT COORDINATES TO NUMPY
    # ========================================================

    particles_x = np.array(
        particles_x
    )

    particles_y = np.array(
        particles_y
    )

    particles_z = np.array(
        particles_z
    )

    # ========================================================
    # 7. SAVE Li DISTRIBUTION
    # ========================================================

    lithium_distribution = {

        # ----------------------------------------------------
        # Depth [cm]
        # ----------------------------------------------------

        "x":
            x_slices,

        # ----------------------------------------------------
        # Theoretical number of primary Li
        # with r = 20 um
        # ----------------------------------------------------

        "N20_theoretical":
            N20_theoretical,

        # ----------------------------------------------------
        # Sampled agglomerate radius [cm]
        # ----------------------------------------------------

        "R_sampled":
            R_sampled_per_x,

        # ----------------------------------------------------
        # Theoretical number of agglomerates
        # ----------------------------------------------------

        "N_agglomerates_theoretical":
            N_agglomerates_theoretical,

        # ----------------------------------------------------
        # Actually generated agglomerates
        # ----------------------------------------------------

        "N_agglomerates_accepted":
            N_agglomerates_accepted
    }

    # ========================================================
    # 8. SUMMARY
    # ========================================================

    print("")

    print(
        "Total theoretical primary Li (R=20 um) =",
        np.sum(
            N20_theoretical
        )
    )

    print(
        "Total theoretical agglomerates =",
        np.sum(
            N_agglomerates_theoretical
        )
    )

    print(
        "Total accepted agglomerates =",
        total_particles
    )

    # ========================================================
    # 9. RETURN
    # ========================================================

    return (
        cells,
        saturation_depth,
        particles_x,
        particles_y,
        particles_z,
        lithium_distribution
    )





# ============================================================
# CHARGE TRANSPORT NUMBA
# ============================================================

@njit
def charge_transport_numba(

        x0,

        N,

        particle_x,

        particle_y,

        particle_z,

        N_particles,

        saturation_depth,

        dx,

        Lx,

        Ly,

        Lz,

        Nt,

        sigma,

        FCCD,

        r

):


    collected = 0

    trapped = 0



    for n in range(N):


        x = x0

        y = np.random.random()*Ly

        z = np.random.random()*Lz



        for step in range(Nt):


            x += sigma*np.random.randn()

            y += sigma*np.random.randn()

            z += sigma*np.random.randn()



            # reflective boundaries

            if y < 0:

                y = -y


            if y > Ly:

                y = 2*Ly-y



            if z < 0:

                z = -z


            if z > Lz:

                z = 2*Lz-z




            # electrode
            
            
            if x >= FCCD:

                collected += 1

                break




            # dead layer

            if x <= dx:

                trapped += 1

                break




            # Li trapping

            if x <= saturation_depth:


                if check_trapping_numba(

                    x,

                    y,

                    z,

                    particle_x,

                    particle_y,

                    particle_z,

                    N_particles,

                    r

                ):


                    trapped += 1

                    break



    return collected, trapped



# ============================================================
# CCE SIMULATION
# ============================================================

def simulate_CCE(
        cells,
        saturation_depth,
        particle_x,
        particle_y,
        particle_z,
        args
):


    print("")
    print("Starting CCE simulation...")
    print("")



    N_particles = len(particle_x)



    x_positions = np.arange(
        0,
        args.FCCD  + 3*step,
        step
    )



    CCE_matrix = np.zeros(
        (
            len(x_positions),
            args.N_repeat
        )
    )



    for i, x0 in enumerate(
        tqdm(
            x_positions,
            desc="CCE depth"
        )
    ):


        print(
            "Charge initial position:",
            x0
        )



        for j in range(args.N_repeat):


            collected, trapped = charge_transport_numba(

                x0,

                args.N_charges,

                particle_x,

                particle_y,

                particle_z,

                N_particles,

                saturation_depth,

                dx,

                Lx,

                Ly,

                Lz,

                Nt,

                sigma,

                args.FCCD,

                r_Li

            )



            CCE_matrix[i,j] = (

                collected /

                args.N_charges

            )





    CCE_mean = np.mean(
        CCE_matrix,
        axis=1
    )



    CCE_std = np.std(
        CCE_matrix,
        axis=1
    )



    return (

        x_positions,

        CCE_matrix,

        CCE_mean,

        CCE_std

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
        args,
        params
):


    os.makedirs(
        args.output,
        exist_ok=True
    )



    filename = os.path.join(
        args.output,
        f"curve_{args.seed:06d}.npz"
    )



    np.savez(

        filename,


        # CCE

        x=x_positions,

        CCE=CCE_matrix,

        CCE_mean=CCE_mean,

        CCE_std=CCE_std,



        # Li distribution

        particle_x=particle_x,

        particle_y=particle_y,

        particle_z=particle_z,



        # Detector

        Lx=Lx,

        Ly=Ly,

        Lz=Lz,

        FCCD_cm=args.FCCD,



        # Lithium parameters

        alpha=params["alpha"],

        T_ann=params["T_ann"],

        t_ann=params["t_ann"],

        Ns=params["Ns"],

        D_Li=params["D_Li"],

        Nd_saturation=Nd_saturation,

        saturation_depth=saturation_depth,



        # Simulation parameters

        N_charges=args.N_charges,

        N_repeat=args.N_repeat,

        seed=args.seed,

        D=D,

        dt=dt,

        sigma=sigma

    )



    print("")
    print("Saved:")
    print(filename)
    print("")





# ============================================================
# MAIN
# ============================================================

def main():



    params = update_parameters(args)



    print("--------------------------------")
    print("CCE simulation")
    print("--------------------------------")
    print("seed =", args.seed)
    print("T_ann =", params["T_ann"])
    print("t_ann =", params["t_ann"])
    print("alpha =", params["alpha"])
    print("--------------------------------")



    (
        cells,

        saturation_depth,

        particle_x,

        particle_y,

        particle_z

    ) = generate_lithium(params)





    (
        x_positions,

        CCE_matrix,

        CCE_mean,

        CCE_std

    ) = simulate_CCE(

        cells,

        saturation_depth,

        particle_x,

        particle_y,

        particle_z,

        args

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

        args,

        params

    )





# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()