import numpy as np
from shapely.geometry import Polygon
import pyproj
# import rospy


def clamp(val, max):
    if val > max:
        return max
    elif val < -max:
        return -max
    return val


def heading_error(heading, goal, dtype="rad"):
    """Resolves cases where the error calculation surrounds the boundaries of pi and -pi.

    If heading is equal to pi and the goal is equal to -pi, without this function the result
    would be 2pi. With this function the result would be 0 as intended, as in orientation pi =
    -pi.

    Args:
        heading: The heading of the Warthog.
        goal: The target heading.

    Returns:
        The error between heading and goal with boundary condition compensation.
    """
    if dtype == "rad":
        error = goal - heading
    elif dtype == "deg":
        error = np.deg2rad(goal) - np.deg2rad(heading)
    if error > np.pi:
        error -= 2 * np.pi
    elif error < -np.pi:
        error += 2 * np.pi
    if dtype == "rad":
        return error
    elif dtype == "deg":
        return np.rad2deg(error)


def point_distance(a, b):
    return np.sqrt(np.power(a[0] - b[0], 2) + np.power(a[1] - b[1], 2))


def wrap_pi(ang, dtype="rad"):
    """Wraps a heading that is outside the bounds of -pi to pi to be within those bounds."""
    if dtype == "rad":
        return (ang + np.pi) % (2 * np.pi) - np.pi
    elif dtype == "deg":
        return np.rad2deg((np.deg2rad(ang) + np.pi) % (2 * np.pi) - np.pi)


def point_angle(a, b, dtype="rad", otype="rot"):
    del_y = (b[1] - a[1]) * -1
    del_x = b[0] - a[0]
    if otype == "rot":
        rot = np.pi / 2
    else:
        rot = 0
    ans = wrap_pi(np.arctan2(del_y, del_x) + rot)
    if dtype == "rad":
        return ans
    elif dtype == "deg":
        return np.rad2deg(ans)


def angle_minmax_conversion(a, conv="one2three", dtype="deg"):
    if dtype == "deg":
        a = np.deg2rad(a)
    if conv == "one2three":
        if a >= 0:
            ret = a
        else:
            ret = a + (2 * np.pi)
    elif conv == "three2one":
        if a <= np.pi/2:
            ret = a
        else:
            ret = a - (2 * np.pi)
    if dtype == "deg":
        return np.rad2deg(ret)
    return ret


def mat2d_rot(vector, theta):
    rot = np.array([[np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]])
    return np.dot(rot, np.array(vector))


def cart_to_polar(point, dtype="rad", otype="rot"):
    r = np.sqrt(point[0]**2 + point[1]**2)
    phi = np.arctan2(point[1], point[0])
    if otype == "rot":
        phi = wrap_pi(-phi + np.pi / 2)
    if dtype == "deg":
        phi = np.rad2deg(phi)
    return [r, phi]


def polar_to_cart(point, otype="rot"):
    if otype == "rot":
        point[1] = wrap_pi(-point[1] + np.pi/2)
    return np.array([point[0] * np.cos(point[1]), point[0] * np.sin(point[1])])


def line_length(a, b):
    return np.sqrt(np.power(a[0] - b[0], 2) + np.power(a[1] - b[1], 2))


def line_intersection(a, b, c, d):
    determinant = (a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0])
    x = ((a[0] * b[1] - a[1] * b[0]) * (c[0] - d[0]) -
         (a[0] - b[0]) * (c[0] * d[1] - c[1] * d[0])) / determinant
    y = ((a[0] * b[1] - a[1] * b[0]) * (c[1] - d[1]) -
         (a[1] - b[1]) * (c[0] * d[1] - c[1] * d[0])) / determinant
    return np.array([x, y])


def golden_section_search(time_to_check, func, start_time=0., tol=0.01):
    golden = (np.sqrt(5) - 1) / 2  # 1 / phi
    golden2 = (3 - np.sqrt(5)) / 2  # 1 / phi^2
    start = start_time
    end = start_time + time_to_check
    (start, end) = (min(start, end), max(start, end))
    dif = end - start

    # Required steps to achieve tolerance
    n = int(np.ceil(np.log(tol / dif) / np.log(golden)))

    pos1 = start + golden2 * dif
    pos2 = start + golden * dif
    res1 = func(pos1)
    res2 = func(pos2)

    for k in range(n-1):
        if res1 < res2:
            end = pos2
            pos2 = pos1
            res2 = res1
            dif = golden * dif
            pos1 = start + golden2 * dif
            res1 = func(pos1)
        else:
            start = pos1
            pos1 = pos2
            res1 = res2
            dif = golden * dif
            pos2 = start + golden * dif
            res2 = func(pos2)

    if res1 < res2:
        ans = (pos2 - start)/2 + start
    else:
        ans = (pos1 - end)/2 + end

    return round(func(ans), 3), ans


def polygon_collision(a_bounds, b_bounds):
    a_poly = Polygon(a_bounds)
    b_poly = Polygon(b_bounds)
    return a_poly.intersects(b_poly)


def find_corners(shape, pos, bearing):
    fl = mat2d_rot([-shape[0] / 2, shape[1] / 2], np.deg2rad(bearing)) + pos
    fr = mat2d_rot([shape[0] / 2, shape[1] / 2], np.deg2rad(bearing)) + pos
    bl = mat2d_rot([-shape[0] / 2, -shape[1] / 2], np.deg2rad(bearing)) + pos
    br = mat2d_rot([shape[0] / 2, -shape[1] / 2], np.deg2rad(bearing)) + pos
    return [fl, fr, br, bl]


def unit_vector(vec):
    return np.array(vec) / np.linalg.norm(vec)


def cpa(os, ts, time_to_check, s_type="r"):
    if s_type == "g":
        geod  = pyproj.Geod(ellps='WGS84')
    def cpa_calc(t, ctype="r"):
        if s_type == "g":
            os_lon, os_lat, os_ba = geod.fwd(os[1][0], os[1][1], os[3], os[2] * t)
            ts_lon, ts_lat, ts_ba = geod.fwd(ts[1][0], ts[1][1], ts[3], ts[2] * t)
            distance = geod.line_length([os_lon, ts_lon], [os_lat, ts_lat])
            return distance
        os_cpa_pos = [os[1][0] + (os[2] * np.sin(np.deg2rad(os[3])) * t),
                      os[1][1] + (os[2] * np.cos(np.deg2rad(os[3])) * t)]
        ts_cpa_pos = [ts[1][0] + (ts[2] * np.sin(np.deg2rad(ts[3])) * t),
                      ts[1][1] + (ts[2] * np.cos(np.deg2rad(ts[3])) * t)]
        if ctype == "r":
            return point_distance(os_cpa_pos, ts_cpa_pos)
        elif ctype == "p":
            return os_cpa_pos, ts_cpa_pos

    dcpa, tcpa = golden_section_search(time_to_check, cpa_calc)
    return dcpa, tcpa

if __name__ == "__main__":
    print(cpa([0, [150, 10], 10, 0],
        [1, [150.001, 10.001], 10, 180],
        60,
        "g"))