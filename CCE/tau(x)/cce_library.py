"""
cce_library_fast.py

Monte Carlo library for HPGe Charge Collection Efficiency (CCE)

Optimized version:
- numba accelerated propagation
- precomputed trapping lifetime
- optional history saving

Physics unchanged.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc
from tqdm import tqdm

from numba import njit



# ============================================================
# Data containers
# ============================================================

@dataclass
class CCELibrary:

    depth_mm: np.ndarray
    cce: np.ndarray
    response: np.ndarray

    fccd_mm: float
    alpha: float
    dx_um: float



@dataclass
class ChargeHistory:

    depth_mm: np.ndarray
    lifetime_ns: np.ndarray
    collected: np.ndarray
    trapped: np.ndarray



# ============================================================
# Physics constants
# ============================================================

D_CM2_S = 28.9

D_MM2_NS = D_CM2_S * 100.0 / 1e9



t_ann = 2 * 60 * 60

T_ann = 473.0

D_Li = 7.66e-9



m0 = 9.11e-31

m_eff = 0.21 * m0

kB = 1.38e-23

T_diff = 90.0



r_Li_cm = 0.002



v_th = np.sqrt(
    3*kB*T_diff/m_eff
) * 100



sigma_trap = np.pi*r_Li_cm**2



pref_tau = 1.0 / (
    v_th*sigma_trap
)



Ns = 10**(
    21.27 - 2610/T_ann
)



# ============================================================
# Trapping
# ============================================================


def tau_value(depth_mm, alpha):

    if alpha == 0:
        return np.inf


    x_cm = depth_mm / 10.0


    Nd = Ns * erfc(
        x_cm /
        (2*np.sqrt(D_Li*t_ann))
    )


    Nd_sat = 1e14


    return (
        pref_tau /
        (alpha*max(Nd-Nd_sat,1.0))
        *1e9
    )



def make_tau_table(
        fccd_mm,
        alpha,
        dx_mm
):

    x = np.arange(
        0,
        fccd_mm+dx_mm,
        dx_mm
    )


    tau = np.zeros_like(x)


    for i in range(len(x)):

        tau[i] = tau_value(
            x[i],
            alpha
        )


    return x,tau



# ============================================================
# Numba propagation
# ============================================================


@njit
def propagate_3D_numba(

        x0_mm,

        fccd_mm,

        y_size_mm,

        z_size_mm,

        dt_ns,

        tmax_ns,

        tau_table,

        tau_step,

        rdr_mm
):


    step_sigma = np.sqrt(
        2*D_MM2_NS*dt_ns
    )


    x = x0_mm

    y = y_size_mm/2

    z = z_size_mm/2


    t = 0.0



    while t < tmax_ns:



        # diffusion 3D

        x += step_sigma*np.random.normal()

        y += step_sigma*np.random.normal()

        z += step_sigma*np.random.normal()



        # dead layer

        if x <= 0:

            return False,False,t



        # electrode

        if x >= fccd_mm:

            return True,False,t



        # reflection y

        if y < 0:

            y=-y

        elif y > y_size_mm:

            y=2*y_size_mm-y



        # reflection z

        if z < 0:

            z=-z

        elif z > z_size_mm:

            z=2*z_size_mm-z



        # --------------------------
        # trapping
        # --------------------------

        idx=int(x/tau_step)


        if idx >= len(tau_table):

            idx=len(tau_table)-1


        tau=tau_table[idx]


        if tau < 1e-20:

            prob=1.0

        else:

            prob=1.0-np.exp(
                -dt_ns/tau
            )



        if np.random.random() < prob:

            return False,True,t



        t += dt_ns



    return False,False,t



# ============================================================
# Build library
# ============================================================


def build_library(

        fccd_mm=1.0,

        alpha=5e-9,

        dx_um=10,

        ncharges=1000,

        N_repeat=5,

        dt_ns=1.0,

        tmax_ns=100000.0,

        y_size_mm=2.0,

        z_size_mm=2.0,

        seed=1234,

        save_history=False

):


    np.random.seed(seed)



    dx_mm = dx_um/1000.0



    depths = np.arange(

        dx_mm,

        fccd_mm+0.5*dx_mm,

        dx_mm

    )



    nt = int(
        tmax_ns/dt_ns
    )+1



    cce = np.zeros(

        (
            len(depths),
            N_repeat
        )

    )



    response = np.zeros(

        (
            nt,
            len(depths)
        )

    )



    # -----------------------------
    # tau lookup table
    # -----------------------------

    tau_x, tau_table = make_tau_table(

        fccd_mm,

        alpha,

        dx_mm

    )



    tau_step = dx_mm



    # history containers

    hist_depth = []

    hist_time = []

    hist_coll = []

    hist_trap = []




    for i,d0 in tqdm(

            enumerate(depths),

            total=len(depths)

    ):



        response_rep = np.zeros(

            (
                nt,
                N_repeat
            )

        )



        for r in range(N_repeat):


            # seed indipendente

            np.random.seed(

                seed +

                i*100000 +

                r

            )



            collected = 0



            counts=np.zeros(nt)



            for c in range(ncharges):


                ok,trapped,t = propagate_3D_numba(

                    d0,

                    fccd_mm,

                    y_size_mm,

                    z_size_mm,

                    dt_ns,

                    tmax_ns,

                    tau_table,

                    tau_step,

                    0.26

                )



                if save_history:


                    hist_depth.append(d0)

                    hist_time.append(t)

                    hist_coll.append(ok)

                    hist_trap.append(trapped)




                if ok:


                    collected += 1



                    idx=min(

                        int(round(t/dt_ns)),

                        nt-1

                    )


                    counts[idx]+=1




            cce[i,r]=(

                collected /

                ncharges

            )



            response_rep[:,r]=np.cumsum(

                counts/ncharges

            )




        response[:,i]=np.mean(

            response_rep,

            axis=1

        )





    lib=CCELibrary(

        depth_mm=depths,

        cce=cce,

        response=response,

        fccd_mm=fccd_mm,

        alpha=alpha,

        dx_um=dx_um

    )



    if save_history:


        hist=ChargeHistory(

            depth_mm=np.asarray(hist_depth),

            lifetime_ns=np.asarray(hist_time),

            collected=np.asarray(hist_coll),

            trapped=np.asarray(hist_trap)

        )


    else:


        hist=None



    return lib,hist





# ============================================================
# Save example
# ============================================================


if __name__=="__main__":


    lib,hist=build_library(

        fccd_mm=1.0,

        alpha=5e-9,

        dx_um=10,

        ncharges=1000,

        N_repeat=3,

        save_history=True

    )



    np.savez(

        "CCE_library_fast.npz",

        depth_mm=lib.depth_mm,

        cce=lib.cce,

        response=lib.response,

        fccd_mm=lib.fccd_mm,

        alpha=lib.alpha,

        dx_um=lib.dx_um

    )



    if hist is not None:


        np.savez(

            "Charge_history_fast.npz",

            depth_mm=hist.depth_mm,

            lifetime_ns=hist.lifetime_ns,

            collected=hist.collected,

            trapped=hist.trapped

        )



    print("Done.")