import numpy as np


def pixel_to_axial(x, y, size=9.5, rotation='FLAT_TOP'):
    if rotation == 'FLAT_TOP':
        q = x * 2 / 3 / size
        r = (-x / 3 + np.sqrt(3) / 3 * y) / size
    elif rotation == 'POINTY_TOP':
        q = (x * np.sqrt(3) / 3 - y / 3) / size
        r = y * 2 / 3 / size
    return q, r


def axial_to_cube(q, r):
    cx = q
    cz = r
    cy = - cx - cz
    return cx, cy, cz


def cube_to_axial(cx, cy, cz):
    return cx, cz


def hex_round(cx, cy, cz):
    q = np.floor(np.round(cx)).astype(np.int)
    r = np.floor(np.round(cy)).astype(np.int)
    s = np.floor(np.round(cz)).astype(np.int)
    q_diff = abs(q - cx)
    r_diff = abs(r - cy)
    s_diff = abs(s - cz)

    m = np.logical_and(q_diff > r_diff, q_diff > s_diff)
    q[m] = -r[m] - s[m]
    m2 = np.logical_and(r_diff > s_diff, ~m)
    r[m2] = -q[m2] - s[m2]

    m3 = np.logical_and(~m, ~m2)
    s[m3] = -q[m3] - r[m3]
    return s, q, r


def pixel_to_hex(x, y, size=9.5, rotation='FLAT_TOP'):
    q, r = pixel_to_axial(x, y, size=size, rotation=rotation)
    cx, cy, cz = axial_to_cube(q, r)
    cx, cy, cz = hex_round(cx, cy, cz)
    return cube_to_axial(cx, cy, cz)
