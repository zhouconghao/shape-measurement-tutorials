# %%
# start add noise to the image
# what is the relation between error in m vs the noise level applied
# pixel noise vs shape noise (level of 0.2)

# read Dodelson: 1. overview, section 2,4, read section 6,7,8 + magnefication, possibly 9

import sys

sys.path.append(
    "/global/cfs/projectdirs/des/zhou/lsst_shear/shape-measurement-tutorials/src"
)
import galsim, ngmix
import numpy as np
import matplotlib.pyplot as plt
import mdet_meas_tools as mmt
from sim_func import sim_func
import logging

log = logging.getLogger("measure_bias_to_noise")

pixel_noise_std_range = 10**np.linspace(-8, -3, 10)
size_noise_std_range = 10**np.linspace(-8, -1, 10)
shape_noise_std_range = 10**np.linspace(-8, -1, 10)


def plot_m_and_c(pixel_noise_std_range, shape_noise_std_range,
                 size_noise_std_range, pixel_noise_std_list,
                 shape_noise_std_list, size_noise_std_list):

    plt.figure(figsize=(10, 15))

    plt.subplot(3, 2, 1)
    plt.plot(pixel_noise_std_range, [i[0] for i in pixel_noise_std_list])
    plt.xlabel('pixel noise std')
    plt.ylabel('std of m [1e-3]')
    plt.xscale('log')
    plt.yscale('log')
    plt.subplot(3, 2, 2)
    plt.plot(pixel_noise_std_range, [i[1] for i in pixel_noise_std_list])
    plt.xlabel('pixel noise std')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('std of c [1e-5]')

    plt.subplot(3, 2, 3)
    plt.plot(shape_noise_std_range, [i[0] for i in shape_noise_std_list])
    plt.xlabel('shape noise std')
    plt.ylabel('std of m [1e-3]')
    plt.subplot(3, 2, 4)
    plt.plot(shape_noise_std_range, [i[1] for i in shape_noise_std_list])
    plt.xlabel('shape noise std')
    plt.ylabel('std of c [1e-5]')

    plt.subplot(3, 2, 5)
    plt.plot(size_noise_std_range, [i[0] for i in size_noise_std_list])
    plt.xlabel('size noise std')
    plt.ylabel('std of m [1e-3]')
    plt.subplot(3, 2, 6)
    plt.plot(size_noise_std_range, [i[1] for i in size_noise_std_list])
    plt.xlabel('size noise std')
    plt.ylabel('std of c [1e-5]')
    plt.show()


def bias_p_or_m(use_p, use_m, plot=False):

    pixel_noise_std_list = []
    for pixel_noise_std_ in pixel_noise_std_range:
        print(f"{pixel_noise_std_=:.2e}")
        pdata, mdata, m, msd, c, csd, R11 = mmt.run_mdet_sims(
            sim_func=sim_func,
            sim_kwargs={
                'psf_fwhm': 0.8,
                'pixel_noise_std': pixel_noise_std_,
                'size_noise_std': 0,
                'shape_noise_std': 0
            },
            seed=123,
            n_sims=10,
            use_p=use_p,
            use_m=use_m)
        pixel_noise_std_list.append((msd, csd))
        res = mmt.estimate_m_and_c(pdata, mdata, use_m=use_m, use_p=use_p)

    size_noise_std_list = []
    for size_noise_std_ in size_noise_std_range:
        print(f"{size_noise_std_=}")
        pdata, mdata, m, msd, c, csd, R11 = mmt.run_mdet_sims(
            sim_func=sim_func,
            sim_kwargs={
                'psf_fwhm': 0.8,
                'pixel_noise_std': 1e-5,
                'size_noise_std': size_noise_std_,
                'shape_noise_std': 0
            },
            seed=123,
            n_sims=10,
            use_p=use_p,
            use_m=use_m)
        size_noise_std_list.append((msd, csd))
        # res = mmt.estimate_m_and_c(pdata, mdata, use_m=use_m, use_p=use_p)

    shape_noise_std_list = []
    for shape_noise_std_ in shape_noise_std_range:
        print(f"{shape_noise_std_=}")
        pdata, mdata, m, msd, c, csd, R11 = mmt.run_mdet_sims(
            sim_func=sim_func,
            sim_kwargs={
                'psf_fwhm': 0.8,
                'pixel_noise_std': 1e-5,
                'shape_noise_std': shape_noise_std_
            },
            seed=123,
            n_sims=10,
            use_p=use_p,
            use_m=use_m)
        shape_noise_std_list.append((msd, csd))
        # res = mmt.estimate_m_and_c(pdata, mdata, use_m=use_m, use_p=use_p)
    if plot:
        plot_m_and_c(pixel_noise_std_range, shape_noise_std_range,
                     size_noise_std_range, pixel_noise_std_list,
                     shape_noise_std_list, size_noise_std_list)

    return (pixel_noise_std_list, shape_noise_std_list, size_noise_std_list)


def overplot_p_m(pixel_list_p, pixel_list_m, pixel_list_pm, shape_list_p,
                 shape_list_m, shape_list_pm, size_list_p, size_list_m,
                 size_list_pm):

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    axes[0, 0].plot(pixel_noise_std_range, [i[0] for i in pixel_list_p],
                    label="p")
    axes[0, 0].plot(pixel_noise_std_range, [i[0] for i in pixel_list_m],
                    label="m")
    axes[0, 0].plot(pixel_noise_std_range, [i[0] for i in pixel_list_pm],
                    label="pm")
    axes[0, 0].set_xlabel('pixel noise std')
    axes[0, 0].set_ylabel('std of m [1e-3]')

    axes[0, 1].plot(pixel_noise_std_range, [i[1] for i in pixel_list_p],
                    label="p")
    axes[0, 1].plot(pixel_noise_std_range, [i[1] for i in pixel_list_m],
                    label="m")
    axes[0, 1].plot(pixel_noise_std_range, [i[1] for i in pixel_list_pm],
                    label="pm")
    axes[0, 1].set_xlabel('pixel noise std')
    axes[0, 1].set_ylabel('std of c [1e-5]')

    axes[1, 0].plot(shape_noise_std_range, [i[0] for i in shape_list_p],
                    label="p")
    axes[1, 0].plot(shape_noise_std_range, [i[0] for i in shape_list_m],
                    label="m")
    axes[1, 0].plot(shape_noise_std_range, [i[0] for i in shape_list_pm],
                    label="pm")
    axes[1, 0].set_xlabel('shape noise std')
    axes[1, 0].set_ylabel('std of m [1e-3]')

    axes[1, 1].plot(shape_noise_std_range, [i[1] for i in shape_list_p],
                    label="p")
    axes[1, 1].plot(shape_noise_std_range, [i[1] for i in shape_list_m],
                    label="m")
    axes[1, 1].plot(shape_noise_std_range, [i[1] for i in shape_list_pm],
                    label="pm")
    axes[1, 1].set_xlabel('shape noise std')
    axes[1, 1].set_ylabel('std of c [1e-5]')

    axes[2, 0].plot(size_noise_std_range, [i[0] for i in pixel_list_p],
                    label="p")
    axes[2, 0].plot(size_noise_std_range, [i[0] for i in pixel_list_m],
                    label="m")
    axes[2, 0].plot(size_noise_std_range, [i[0] for i in pixel_list_pm],
                    label="pm")
    axes[2, 0].set_xlabel('size noise std')
    axes[2, 0].set_ylabel('std of m [1e-3]')

    axes[2, 1].plot(size_noise_std_range, [i[1] for i in size_list_p],
                    label="p")
    axes[2, 1].plot(size_noise_std_range, [i[1] for i in size_list_m],
                    label="m")
    axes[2, 1].plot(size_noise_std_range, [i[1] for i in size_list_pm],
                    label="pm")
    axes[2, 1].set_xlabel('size noise std')
    axes[2, 1].set_ylabel('std of c [1e-5]')

    for ax in axes.flatten():
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()

    return fig, axes


def bias_sersic_n_and_half_light_radius():

    use_p = True
    use_m = True

    n_range = np.linspace(0.5, 6, 50)
    half_light_radius_range = np.concatenate(
        [np.linspace(0.1, 1, 50),
         np.linspace(0.1, 3, 50)])

    # pixel_noise_std_list = []
    # for pixel_noise_std_ in pixel_noise_std_range:
    #     print(f"{pixel_noise_std_=:.2e}")
    #     pdata, mdata, m, msd, c, csd = mmt.run_mdet_sims(sim_func=sim_func,
    #                                                      sim_kwargs={
    #                                                          'psf_fwhm': 0.8,
    #                                                          'pixel_noise_std':
    #                                                          pixel_noise_std_,
    #                                                          'size_noise_std':
    #                                                          0,
    #                                                          'shape_noise_std':
    #                                                          0
    #                                                      },
    #                                                      seed=123,
    #                                                      n_sims=10,
    #                                                      use_p=use_p,
    #                                                      use_m=use_m)
    #     pixel_noise_std_list.append((msd, csd))
    #     res = mmt.estimate_m_and_c(pdata, mdata, use_m=use_m, use_p=use_p)

    n_range_list = []
    for n_ in n_range:
        print(f"{n_=:.2e}")
        pdata, mdata, m, msd, c, csd, R11 = mmt.run_mdet_sims(
            sim_func=sim_func,
            sim_kwargs={
                'psf_fwhm': 0.8,
                'sersic_n': n_
            },
            seed=123,
            n_sims=10,
            use_p=use_p,
            use_m=use_m)
        n_range_list.append((msd, csd, R11))
        res = mmt.estimate_m_and_c(pdata, mdata, use_m=True, use_p=use_p)

    half_light_radius_range_list = []
    for half_light_radius_ in half_light_radius_range:
        print(f"{half_light_radius_=:.2e}")
        pdata, mdata, m, msd, c, csd, R11 = mmt.run_mdet_sims(
            sim_func=sim_func,
            sim_kwargs={
                'psf_fwhm': 0.8,
                'half_light_radius': half_light_radius_
            },
            seed=123,
            n_sims=10,
            use_p=use_p,
            use_m=use_m)
        half_light_radius_range_list.append((msd, csd, R11))
        res = mmt.estimate_m_and_c(pdata, mdata, use_m=True, use_p=use_p)

    return (n_range, n_range_list), (half_light_radius_range,
                                     half_light_radius_range_list)


def plot_bias_sersic_n_and_half_light_radius(n_range, n_range_list,
                                             half_light_radius_range,
                                             half_light_radius_range_list):

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].scatter(n_range, [i[0] for i in n_range_list], label="m")
    axes[0, 0].set_xlabel('n')
    axes[0, 0].set_ylabel('std of m [1e-3]')

    axes[0, 1].scatter(n_range, [i[1] for i in n_range_list], label="c")
    axes[0, 1].set_xlabel('n')
    axes[0, 1].set_ylabel('std of c [1e-5]')

    axes[1, 0].scatter(half_light_radius_range,
                       [i[0] for i in half_light_radius_range_list],
                       label="m")
    axes[1, 0].set_xlabel('half_light_radius')
    axes[1, 0].set_ylabel('std of m [1e-3]')

    axes[1, 1].scatter(half_light_radius_range,
                       [i[1] for i in half_light_radius_range_list],
                       label="c")
    axes[1, 1].set_xlabel('half_light_radius')
    axes[1, 1].set_ylabel('std of c [1e-5]')

    for ax in axes.flatten():
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()

    return fig, axes


def plot_R11_sesic_n_and_half_light_radius(n_range, n_range_list,
                                           half_light_radius_range,
                                           half_light_radius_range_list):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].scatter(n_range,
                    np.array([i[2] for i in n_range_list]),
                    label="R11")
    axes[0].set_xlabel('Serisc n')
    axes[0].set_ylabel('R11')

    axes[1].scatter(half_light_radius_range,
                    np.array([i[2] for i in half_light_radius_range_list]),
                    label="R11")
    axes[1].set_xlabel('half_light_radius')
    axes[1].set_ylabel('R11')

    for ax in axes.flatten():
        ax.set_xscale("log")
        # ax.set_yscale("log")
        ax.legend()


def R11_in_HLR_bulge_frac_grid(desired_sn, nsims):

    bulge_re = 0.1

    use_p = True
    use_m = True

    bulge_frac_range = np.linspace(0, 1, 21)
    disc_bulge_radius_range = np.linspace(0.1, 2.8, 21)

    R11_array = np.zeros((len(bulge_frac_range), len(disc_bulge_radius_range)))

    for i, bulge_frac in enumerate(bulge_frac_range):
        for j, disc_bulge_radius in enumerate(disc_bulge_radius_range):

            print(
                f"bulge_frac: {bulge_frac:.2f}, disc_bulge_radius:{disc_bulge_radius:.2f}"
            )

            mdet_result = mmt.run_mdet_sims(sim_func=sim_func,
                                            sim_kwargs={
                                                'psf_fwhm': 0.8,
                                                'disc_bulge_radius':
                                                disc_bulge_radius,
                                                'bulge_frac': bulge_frac,
                                                'desired_sn': desired_sn,
                                            },
                                            seed=123,
                                            n_sims=nsims,
                                            use_p=use_p,
                                            use_m=use_m)

            if len(mdet_result) == 2:
                R11 = 0
            else:  #something is cutting the objects
                pdata, mdata, m, msd, c, csd, R11 = mdet_result
                res = mmt.estimate_m_and_c(pdata,
                                           mdata,
                                           use_m=True,
                                           use_p=use_p)
            R11_array[i, j] = R11

    return bulge_frac_range, disc_bulge_radius_range, R11_array


def plot_R11_in_HLR_bulge_frac_grid(bulge_frac_range, disc_bulge_radius_range,
                                    R11_array):

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    im = ax.imshow(R11_array,
                   extent=[
                       bulge_frac_range[0], bulge_frac_range[-1],
                       disc_bulge_radius_range[0], disc_bulge_radius_range[-1]
                   ],
                   origin='lower',
                   aspect='auto')
    ax.set_xlabel('bulge_frac')
    ax.set_ylabel('HLR')
    ax.set_title('R11')
    fig.colorbar(im)

    return fig, ax