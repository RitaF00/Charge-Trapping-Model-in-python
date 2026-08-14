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

'''
parser.add_argument(
    "--seed",
    type=int,
    required=True,
    help="Random seed"
)'''

parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="Random seed"
)


parser.add_argument(
    "--N_charges",
    type=int,
    default=50,
    help="Number of charge carriers"
)


parser.add_argument(
    "--N_repeat",
    type=int,
    default= 50,
    help="Number of repetitions per depth"
)


parser.add_argument(
    "--alpha",
    type=float,
    default=1.6e-11,
    help="Li precipitation coefficient"
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


args = parser.parse_args([])


np.random.seed(args.seed)


# ============================================================
# GEOMETRY (cm)
# ============================================================

Lx = 0.2
Ly = 0.2
Lz = 0.2


dx = 0.001   #10 um

step = 0.005


FCCD_cm = 0.1



# ============================================================
# DIFFUSION
# ============================================================

D = 28.9       # cm2/ns

dt = 1.0       # ns

t_max = 10000

Nt = int(t_max/dt)


sigma = math.sqrt(6*D*dt)*1e-4


# ============================================================
# LITHIUM PARAMETERS
# ============================================================

r_Li = 0.002       # cm

cell_size = r_Li



# ============================================================
# ANNEALING MODEL
# ============================================================

R = 1.98

H = 11800

D0 = 2.5e-3



T_ann = args.T_ann

t_ann = args.t_ann


Ns = 10**(21.27 - 2610/T_ann)


D_Li = D0 * math.exp(
    -H/(R*T_ann)
)



Nd_saturation = 1e14


alpha = args.alpha



# ============================================================
# RANDOM INFO
# ============================================================

print("--------------------------------")
print("CCE simulation")
print("--------------------------------")
print("seed =", args.seed)
print("T_ann =", T_ann)
print("t_ann =", t_ann)
print("alpha =", alpha)
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
    """
    Convert Cartesian coordinates into grid indices
    """


    ix = int(x / cell_size)
    iy = int(y / cell_size)
    iz = int(z / cell_size)


    # safety limits

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

    """
    Check if a new Li precipitate overlaps
    with existing precipitates
    """


    ix, iy, iz = get_index(
        x,
        y,
        z,
        nx,
        ny,
        nz
    )


    # Search neighbouring cells

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

    """
    Create the spatial grid used for
    Li precipitate lookup
    """


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

def calculate_saturation_depth():

    depth_list = np.arange(
        0,
        0.11 + dx,
        dx
    )


    Nd_vals = Ns * erfc(
        depth_list /
        (2*np.sqrt(D_Li*t_ann))
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
# GENERATE LITHIUM DISTRIBUTION
# ============================================================

def generate_lithium():


    print("")
    print("Generating Li distribution...")


    saturation_depth = calculate_saturation_depth()


    print(
        "Saturation depth =",
        saturation_depth*10,
        "mm"
    )



    cells, nx, ny, nz = create_grid()



    particles_x = []
    particles_y = []
    particles_z = []



    x_slices = np.arange(
        0,
        saturation_depth,
        dx
    )


    total_particles = 0



    # volume of a Li precipitate

    V_particle = (
        4/3 *
        math.pi *
        r_Li**3
    )



    for xi in tqdm(
        x_slices,
        desc="Li slices"
    ):



        # Li concentration profile

        Nd_val = Ns * erfc(
            xi /
            (2*np.sqrt(D_Li*t_ann))
        )



        density = (
            alpha *
            max(
                Nd_val-Nd_saturation,
                0
            )
        )



        # expected number of particles

        V_slice = dx*Ly*Lz


        lam = density*V_slice



        # maximum geometrical packing
        pf = 0.64  # --> packing factor : random close packing of hard spheres
        Nmax = int(
            pf * V_slice/V_particle
        )



        N_i = truncated_poisson(
            lam,
            Nmax
        )



        accepted = 0


        attempts = 0


        max_attempts = max(
            100,
            N_i*200
        )



        while (
            accepted < N_i and
            attempts < max_attempts
        ):


            attempts += 1



            x = (
                xi +
                np.random.random()*dx
            )


            y = (
                np.random.random()*Ly
            )


            z = (
                np.random.random()*Lz
            )



            if not is_overlapping(
                x,
                y,
                z,
                r_Li,
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



                p = LiParticle(
                    x,
                    y,
                    z,
                    r_Li
                )


                cells[ix][iy][iz].append(p)



                particles_x.append(x)

                particles_y.append(y)

                particles_z.append(z)



                accepted += 1

                total_particles += 1



    print(
        "Total Li particles =",
        total_particles
    )



    particles_x = np.array(
        particles_x
    )

    particles_y = np.array(
        particles_y
    )

    particles_z = np.array(
        particles_z
    )



    return (
        cells,
        saturation_depth,
        particles_x,
        particles_y,
        particles_z
    )






# ============================================================
# CHARGE TRANSPORT + TRAPPING
# ============================================================
'''
def check_trapping(
        x,
        y,
        z,
        cells,
        nx,
        ny,
        nz
):
    """
    Check if a charge is captured by a Li precipitate
    """

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


                    if distance2 < r_Li**2:

                        return True



    return False
'''

@njit
def check_trapping_numba(
        x,
        y,
        z,
        particle_x,
        particle_y,
        particle_z,
        N_particles,
        r
):

    r2 = r*r


    for i in range(N_particles):

        dx = x - particle_x[i]
        dy = y - particle_y[i]
        dz = z - particle_z[i]


        if dx*dx + dy*dy + dz*dz < r2:

            return True


    return False


# ============================================================
# SINGLE CCE TRANSPORT
# ============================================================

def charge_transport(
        x_start,
        N,
        cells,
        saturation_depth
):


    nx = len(cells)

    ny = len(cells[0])

    nz = len(cells[0][0])



    collected = 0

    trapped = 0



    for i in range(N):


        # --------------------------------
        # Initial position
        # --------------------------------

        x = x_start

        y = np.random.random()*Ly

        z = np.random.random()*Lz



        is_trapped = False



        # --------------------------------
        # Diffusion random walk
        # --------------------------------

        for step in range(Nt):


            x += sigma*np.random.randn()

            y += sigma*np.random.randn()

            z += sigma*np.random.randn()



            # boundary conditions
            y = np.clip(y,0,Ly)
            z = np.clip(z,0,Lz)



            # --------------------------------
            # collected electrode
            # --------------------------------

            if x >= FCCD_cm:

                collected += 1

                break



            # --------------------------------
            # dead layer
            # --------------------------------

            if x <= dx:

                trapped += 1

                break



            # --------------------------------
            # Li trapping region
            # --------------------------------

            if x <= saturation_depth:


                if check_trapping(
                    x,
                    y,
                    z,
                    cells,
                    nx,
                    ny,
                    nz
                ):

                    trapped += 1

                    is_trapped = True

                    break



        # if not collected and not trapped:
        # lost carrier

    return collected, trapped

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

        FCCD_cm,

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



            # -------------------------
            # reflective boundaries y,z
            # -------------------------

            if y < 0:

                y = -y

            if y > Ly:

                y = 2*Ly-y



            if z < 0:

                z = -z

            if z > Lz:

                z = 2*Lz-z



            # -------------------------
            # electrodes
            # -------------------------

            if x >= FCCD_cm:

                collected += 1
                break



            # -------------------------
            # dead layer
            # -------------------------

            if x <= dx:

                trapped += 1
                break



            # -------------------------
            # Li trapping
            # -------------------------

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
        particle_z):


    print("")
    print("Starting CCE simulation...")
    print("")


    N_particles = len(particle_x)
    # Depth points

    x_positions = np.arange(
        0,
        FCCD_cm + step,
        step
    )



    CCE_matrix = np.zeros(
        (
            len(x_positions),
            args.N_repeat
        )
    )



    # Loop over interaction depth

    for i, x0 in enumerate(
        tqdm(
            x_positions,
            desc="CCE depth"
        )
    ):


        print("Charge intial position:", x0)



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
                                FCCD_cm,                            
                                r_Li )


            CCE_matrix[i,j] = (
                collected /
                args.N_charges
            )



    # Average curve

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


"""

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
        particle_z
):


    # create output directory

    os.makedirs(
        args.output,
        exist_ok=True
    )


    filename = os.path.join(
        args.output,f"curve_{args.seed:06d}.npz")


    np.savez(
        filename,


        # -------------------------
        # CCE data
        # -------------------------

        x=x_positions,

        CCE=CCE_matrix,

        CCE_mean=CCE_mean,

        CCE_std=CCE_std,



        # -------------------------
        # Lithium distribution
        # -------------------------

        particle_x=particle_x,

        particle_y=particle_y,

        particle_z=particle_z,



        # -------------------------
        # Detector parameters
        # -------------------------

        Lx=Lx,

        Ly=Ly,

        Lz=Lz,

        FCCD_cm=FCCD_cm,



        # -------------------------
        # Lithium parameters
        # -------------------------

        alpha=alpha,

        T_ann=T_ann,

        t_ann=t_ann,

        Ns=Ns,

        D_Li=D_Li,

        Nd_saturation=Nd_saturation,

        saturation_depth=saturation_depth,



        # -------------------------
        # Simulation parameters
        # -------------------------

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


    # -------------------------
    # Generate lithium
    # -------------------------

    (
        cells,
        saturation_depth,
        particle_x,
        particle_y,
        particle_z

    ) = generate_lithium()



    # -------------------------
    # Simulate CCE
    # -------------------------

    (
        x_positions,
        CCE_matrix,
        CCE_mean,
        CCE_std

    ) = simulate_CCE(
        cells,
        saturation_depth
    )



    # -------------------------
    # Save library curve
    # -------------------------

    save_results(
        x_positions,
        CCE_matrix,
        CCE_mean,
        CCE_std,
        saturation_depth,
        particle_x,
        particle_y,
        particle_z
    )



# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()"""