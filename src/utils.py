import numpy as np


def get_moments(im, x, y, verbose=False):

    # M = skimage.measure.moments_central(im,order=2)

    M_0 = np.sum(im)

    M_x = np.sum(x * im) / M_0
    M_y = np.sum(y * im) / M_0

    M_xx = np.sum((x - M_x)**2 * im) / M_0
    M_yy = np.sum((y - M_y)**2 * im) / M_0
    M_xy = np.sum((x - M_x) * (y - M_y) * im) / M_0

    return ((M_x, M_y, M_xx, M_xy, M_yy))


def get_elipticity(M_xx, M_xy, M_yy):

    M_r = M_xx + M_yy
    M_p = M_xx - M_yy
    M_x = 2 * M_xy

    e1 = M_p / M_r
    e2 = M_x / M_r

    return (e1, e2)


def get_x_y(im, scale):
    xmin_zero = im.bounds.xmin - 1
    xmax_zero = im.bounds.xmax - 1
    ymin_zero = im.bounds.ymin - 1
    ymax_zero = im.bounds.ymax - 1

    x, y = np.meshgrid(np.arange(xmin_zero, xmax_zero + 1),
                       np.arange(ymin_zero, ymax_zero + 1))
    x, y = x * scale, y * scale

    return (x, y)
