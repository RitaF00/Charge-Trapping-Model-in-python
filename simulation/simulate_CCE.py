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

def generate_lithium(params):


    print("")
    print("Generating Li distribution...")
    print(f"alpha {params['alpha']:.2e}")


    saturation_depth = calculate_saturation_depth(params)


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

        Nd_val = params["Ns"] * erfc(
            xi /
            (
                2*np.sqrt(
                    params["D_Li"] *
                    params["t_ann"]
                )
            )
        )



        density = (

            params["alpha"] *

            max(
                Nd_val - Nd_saturation,
                0
            )
        )



        V_slice = dx*Ly*Lz


        lam = density*V_slice



        # maximum packing

        pf = 0.64


        Nmax = int(
            pf *
            V_slice /
            V_particle
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
        N_particles,
        r
):


    r2 = r*r



    for i in range(N_particles):


        dx = x - particle_x[i]

        dy = y - particle_y[i]

        dz = z - particle_z[i]



        if (
            dx*dx +
            dy*dy +
            dz*dz
        ) < r2:

            return True



    return False






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
        args.FCCD  + step,
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