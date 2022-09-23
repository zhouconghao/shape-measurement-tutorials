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

pixel_noise_std_range = 10**np.linspace(-8, -3, 10)
size_noise_std_range = 10**np.linspace(-8, -1, 10)
shape_noise_std_range = 10**np.linspace(-8, -1, 10)


def sim_func(*,
             g1,
             g2,
             seed,
             psf_fwhm,
             pixel_noise_std,
             size_noise_std=0,
             shape_noise_std=0):
    # this is an RNG you can use to draw random numbers as needed
    # always use this RNG and not np.random directly
    # doing this makes sure the code is reproducible

    rng = np.random.RandomState(seed=seed)

    size_nse = rng.normal(loc=0, scale=size_noise_std)
    # make an Exponential object in galsim with a half light radius of 0.5
    gal = galsim.Exponential(half_light_radius=0.5 + size_nse)

    g1_noise = rng.normal(loc=0, scale=shape_noise_std)
    g2_noise = rng.normal(loc=0, scale=shape_noise_std)

    gal = gal.shear(g1=g1_noise, g2=g2_noise)

    # make a Gaussian object in galsim with a fwhm of `psf_fwhm`
    psf = galsim.Gaussian(fwhm=psf_fwhm)

    # apply the input shear `g1`, `g2` to the galaxy `gal`

    ### draw shear randomly scale=0.3 for shear
    sheared_gal = gal.shear(g1=g1, g2=g2)

    # here we are going to apply a random shift to the object's center
    dx, dy = 2.0 * (rng.uniform(size=2) - 0.5) * 0.2
    sheared_gal = sheared_gal.shift(dx, dy)

    # convolve the sheared galaxy with the psf
    obj = galsim.Convolve(sheared_gal, psf)

    # render the object and the PSF on odd sized images of 53 pixels on a side with
    # a pixel scale of 0.2
    obj_im = obj.drawImage(scale=0.2, nx=53, ny=53)
    psf_im = psf.drawImage(scale=0.2, nx=53, ny=53)

    # now we are going to add noise to the object image and setup the ngmix data
    cen = (53 - 1) / 2
    nse_sd = pixel_noise_std
    nse = rng.normal(size=obj_im.array.shape, scale=nse_sd)
    nse_im = rng.normal(size=obj_im.array.shape, scale=nse_sd)

    jac = ngmix.jacobian.DiagonalJacobian(scale=0.2,
                                          row=cen + dy / 0.2,
                                          col=cen + dx / 0.2)
    psf_jac = ngmix.jacobian.DiagonalJacobian(scale=0.2, row=cen, col=cen)

    # Transformation between pixel and tangent uv coordinate. It has off diag terms if CCD is rotated.
    # in real data we care about off diag terms

    # we have to add a little noise to the PSf to make things stable
    target_psf_s2n = 500.0
    target_psf_noise = np.sqrt(np.sum(psf_im.array**2)) / target_psf_s2n

    psf_obs = ngmix.Observation(
        image=psf_im.array,
        weight=np.ones_like(psf_im.array) / target_psf_noise**2,
        jacobian=psf_jac,
    )

    # here we build the final observation
    obj_obs = ngmix.Observation(
        image=obj_im.array + nse,
        noise=nse_im,
        weight=np.ones_like(nse_im) / nse_sd**2,
        jacobian=psf_jac,
        bmask=np.zeros_like(nse_im, dtype=np.int32),
        ormask=np.zeros_like(nse_im, dtype=np.int32),
        psf=psf_obs,
    )

    return obj_obs


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
        pdata, mdata, m, msd, c, csd = mmt.run_mdet_sims(sim_func=sim_func,
                                                         sim_kwargs={
                                                             'psf_fwhm': 0.8,
                                                             'pixel_noise_std':
                                                             pixel_noise_std_,
                                                             'size_noise_std':
                                                             0,
                                                             'shape_noise_std':
                                                             0
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
        pdata, mdata, m, msd, c, csd = mmt.run_mdet_sims(sim_func=sim_func,
                                                         sim_kwargs={
                                                             'psf_fwhm': 0.8,
                                                             'pixel_noise_std':
                                                             1e-5,
                                                             'size_noise_std':
                                                             size_noise_std_,
                                                             'shape_noise_std':
                                                             0
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
        pdata, mdata, m, msd, c, csd = mmt.run_mdet_sims(sim_func=sim_func,
                                                         sim_kwargs={
                                                             'psf_fwhm':
                                                             0.8,
                                                             'pixel_noise_std':
                                                             1e-5,
                                                             'shape_noise_std':
                                                             shape_noise_std_
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
