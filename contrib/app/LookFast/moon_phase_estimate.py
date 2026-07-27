from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt

from skyfield.api import load, wgs84
from skyfield import almanac

import contrib.app.LookFast.lookback_tools as lbt


def plot_moon_phase_from_location(
    when=None,
    observer_dms=((34, 57, 44.56), (-106, 30, 34.90), 1756.8672),
    ephemeris_file_path=None,
    ephemeris_filename="de430t.bsp",
    grid_size=800,
    show=True,
):
    """
    Calculate and plot the Moon phase as viewed from a specified observation location.

    Parameters
    ----------
    when : datetime.datetime, str, skyfield.timelib.Time, or None
        Observation time. If None, uses current UTC time.
        If a naive datetime is supplied, it is assumed to be UTC.
        ISO-8601 strings are accepted, e.g. "2026-07-24T04:00:00Z".

    observer_dms : tuple
        Tuple of the form:
            ((lat_deg, lat_min, lat_sec),
             (lon_deg, lon_min, lon_sec),
             elevation_m)

        Longitude west should be negative, as in:
            (-106, 30, 34.90)

    ephemeris_path : str
        Path to the JPL ephemeris file, e.g. "de430t.bsp".

    grid_size : int
        Resolution of the rendered Moon image.

    show : bool
        If True, calls plt.show().

    Returns
    -------
    result : dict
        Dictionary containing computed Moon phase data and matplotlib objects.
    """

    def dms_to_decimal(dms):
        deg, minute, second = dms
        sign = -1.0 if deg < 0 else 1.0
        return sign * (abs(deg) + minute / 60.0 + second / 3600.0)

    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def altaz_to_enu(alt_rad, az_rad):
        """
        Convert altitude/azimuth to local East-North-Up unit vector.

        Skyfield azimuth is measured from North toward East.
        """
        east = np.cos(alt_rad) * np.sin(az_rad)
        north = np.cos(alt_rad) * np.cos(az_rad)
        up = np.sin(alt_rad)
        return np.array([east, north, up], dtype=float)

    def phase_name_from_elongation(elongation_deg):
        """
        Skyfield almanac.moon_phase() returns an angle where approximately:
            0   deg = New Moon
            90  deg = First Quarter
            180 deg = Full Moon
            270 deg = Last Quarter
        """
        e = elongation_deg % 360.0

        if e < 22.5 or e >= 337.5:
            return "New Moon"
        elif e < 67.5:
            return "Waxing Crescent"
        elif e < 112.5:
            return "First Quarter"
        elif e < 157.5:
            return "Waxing Gibbous"
        elif e < 202.5:
            return "Full Moon"
        elif e < 247.5:
            return "Waning Gibbous"
        elif e < 292.5:
            return "Last Quarter"
        else:
            return "Waning Crescent"

    # -------------------------------------------------------------------------
    # Time handling
    # -------------------------------------------------------------------------
    ts = load.timescale()

    if when is None:
        dt = datetime.now(timezone.utc)
        t = ts.from_datetime(dt)
    elif hasattr(when, "tt"):
        # Already a Skyfield Time object
        t = when
        dt = None
    elif isinstance(when, datetime):
        dt = when
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        t = ts.from_datetime(dt)
    elif isinstance(when, str):
        # Accept a common ISO string with trailing Z
        when_clean = when.replace("Z", "+00:00")
        dt = datetime.fromisoformat(when_clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        t = ts.from_datetime(dt)
    else:
        raise TypeError("when must be None, datetime, ISO-8601 string, or Skyfield Time object.")

    # -------------------------------------------------------------------------
    # Observer location
    # -------------------------------------------------------------------------
    lat_dms, lon_dms, elevation_m = observer_dms

    latitude_deg = dms_to_decimal(lat_dms)
    longitude_deg = dms_to_decimal(lon_dms)

    location = wgs84.latlon(latitude_degrees=latitude_deg, longitude_degrees=longitude_deg, elevation_m=elevation_m)

    # -------------------------------------------------------------------------
    # Load ephemeris
    # -------------------------------------------------------------------------
    eph = lbt.load_de430_ephemeris(ephemeris_path=ephemeris_file_path, filename=ephemeris_filename)

    earth = eph["earth"]
    moon = eph["moon"]
    sun = eph["sun"]

    observer = earth + location

    # -------------------------------------------------------------------------
    # Apparent topocentric Moon and Sun positions
    # -------------------------------------------------------------------------
    moon_app = observer.at(t).observe(moon).apparent()
    sun_app = observer.at(t).observe(sun).apparent()

    moon_alt, moon_az, moon_distance = moon_app.altaz()
    sun_alt, sun_az, sun_distance = sun_app.altaz()

    moon_alt_deg = moon_alt.degrees
    moon_az_deg = moon_az.degrees
    sun_alt_deg = sun_alt.degrees
    sun_az_deg = sun_az.degrees

    # -------------------------------------------------------------------------
    # Moon phase angle and illuminated fraction
    # -------------------------------------------------------------------------
    # Elongation-type angle useful for naming the phase.
    elongation_deg = almanac.moon_phase(eph, t).degrees % 360.0
    phase_name = phase_name_from_elongation(elongation_deg)

    # Compute topocentric illuminated fraction geometrically.
    # At the Moon, alpha is the Sun-Moon-Observer angle.
    obs_bary = observer.at(t).position.au
    moon_bary = moon.at(t).position.au
    sun_bary = sun.at(t).position.au

    moon_to_sun = normalize(sun_bary - moon_bary)
    moon_to_observer = normalize(obs_bary - moon_bary)

    cos_alpha = np.clip(np.dot(moon_to_sun, moon_to_observer), -1.0, 1.0)
    alpha_rad = np.arccos(cos_alpha)

    illuminated_fraction = 0.5 * (1.0 + cos_alpha)
    alpha_deg = np.degrees(alpha_rad)

    # -------------------------------------------------------------------------
    # Determine apparent bright-limb direction in the local sky
    # -------------------------------------------------------------------------
    moon_enu = altaz_to_enu(moon_alt.radians, moon_az.radians)
    sun_enu = altaz_to_enu(sun_alt.radians, sun_az.radians)

    # Image y-axis: local zenith projected into plane of the sky at the Moon.
    zenith = np.array([0.0, 0.0, 1.0])
    y_axis = zenith - np.dot(zenith, moon_enu) * moon_enu

    # If Moon is very near zenith, local "up" is poorly defined.
    if np.linalg.norm(y_axis) < 1e-10:
        north = np.array([0.0, 1.0, 0.0])
        y_axis = north - np.dot(north, moon_enu) * moon_enu

    y_axis = normalize(y_axis)

    # Image x-axis: right-hand direction in the viewer's sky plane.
    x_axis = normalize(np.cross(moon_enu, y_axis))

    # Project Sun direction into the sky plane centered on the Moon.
    sun_proj = sun_enu - np.dot(sun_enu, moon_enu) * moon_enu

    if np.linalg.norm(sun_proj) < 1e-12:
        # Near exact full or new Moon, the limb direction is not well-defined.
        bright_dir = np.array([1.0, 0.0])
    else:
        sun_proj = normalize(sun_proj)
        bright_dir = np.array([np.dot(sun_proj, x_axis), np.dot(sun_proj, y_axis)])
        bright_dir = normalize(bright_dir)

    illuminated_vertices = illuminated_moon_polygon(alpha_rad=alpha_rad, bright_dir=bright_dir, n=grid_size)

    # -------------------------------------------------------------------------
    # Render illuminated lunar disk
    # -------------------------------------------------------------------------
    n = grid_size
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(x, y)

    R2 = X**2 + Y**2
    disk = R2 <= 1.0

    Z = np.zeros_like(X)
    Z[disk] = np.sqrt(1.0 - R2[disk])

    # Sun direction in Moon-centered image coordinates.
    # z points toward the observer.
    sx = np.sin(alpha_rad) * bright_dir[0]
    sy = np.sin(alpha_rad) * bright_dir[1]
    sz = np.cos(alpha_rad)

    illumination = X * sx + Y * sy + Z * sz
    lit = disk & (illumination > 0.0)

    rgba = np.zeros((n, n, 4), dtype=float)

    # Dark side
    rgba[disk, 0] = 0.08
    rgba[disk, 1] = 0.08
    rgba[disk, 2] = 0.09
    rgba[disk, 3] = 1.0

    # Lit side
    rgba[lit, 0] = 0.92
    rgba[lit, 1] = 0.90
    rgba[lit, 2] = 0.82
    rgba[lit, 3] = 1.0

    # Transparent outside disk
    rgba[~disk, 3] = 0.0

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    ax.set_facecolor("black")

    ax.imshow(rgba, extent=(-1, 1, -1, 1), origin="lower", interpolation="bilinear")

    limb = plt.Circle((0, 0), 1.0, edgecolor="0.7", facecolor="none", linewidth=1.5)
    ax.add_patch(limb)

    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.axis("off")

    time_label = t.utc_iso()

    title = (
        f"{phase_name}\n"
        f"Illuminated: {illuminated_fraction * 100:.1f}%    "
        f"Phase angle: {alpha_deg:.2f}°\n"
        f"Moon alt/az: {moon_alt_deg:.2f}° / {moon_az_deg:.2f}° UTC {time_label}"
    )

    ax.set_title(title, color="white", fontsize=11)

    # Direction markers
    ax.text(0.0, 1.04, "Zenith", color="white", ha="center", va="bottom", fontsize=9)

    ax.annotate(
        "Sun",
        xy=(0.75 * bright_dir[0], 0.75 * bright_dir[1]),
        xytext=(1.12 * bright_dir[0], 1.12 * bright_dir[1]),
        color="yellow",
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", color="yellow"),
    )

    if moon_alt_deg < 0:
        ax.text(
            0.0,
            -1.16,
            "Warning: Moon is below the local horizon at this time.",
            color="orange",
            ha="center",
            va="top",
            fontsize=9,
        )

    if show:
        plt.show()

    return {
        "time_utc": time_label,
        "latitude_deg": latitude_deg,
        "longitude_deg": longitude_deg,
        "elevation_m": elevation_m,
        "phase_name": phase_name,
        "elongation_deg": elongation_deg,
        "phase_angle_deg": alpha_deg,
        "phase_angle_rad": alpha_rad,
        "bright_dir": bright_dir,
        "illuminated_vertices": illuminated_vertices,
        "illuminated_fraction": illuminated_fraction,
        "moon_altitude_deg": moon_alt_deg,
        "moon_azimuth_deg": moon_az_deg,
        "sun_altitude_deg": sun_alt_deg,
        "sun_azimuth_deg": sun_az_deg,
        "moon_enu": moon_enu,
        "sky_x_axis_enu": x_axis,
        "sky_y_axis_enu": y_axis,
        "figure": fig,
        "axes": ax,
    }


import numpy as np


def illuminated_moon_polygon(alpha_rad, bright_dir=(1.0, 0.0), n=512):
    """
    Return polygon vertices for the illuminated portion of the apparent lunar disk.

    Parameters
    ----------
    alpha_rad : float
        Moon phase angle in radians.
        0 means full Moon, pi/2 means quarter Moon, pi means new Moon.

    bright_dir : array-like, shape (2,)
        Unit vector in image coordinates pointing toward the Sun.

    n : int
        Number of samples used along the terminator and limb.

    Returns
    -------
    vertices_xy : ndarray, shape (M, 2)
        Polygon vertices in normalized Moon image coordinates.
        The full Moon disk has radius 1.
    """

    bright_dir = np.asarray(bright_dir, dtype=float)
    bright_dir = bright_dir / np.linalg.norm(bright_dir)

    # Perpendicular direction in the image plane.
    perp_dir = np.array([-bright_dir[1], bright_dir[0]])

    # Handle nearly full Moon.
    if alpha_rad < 1.0e-8:
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        return np.column_stack([x, y])

    # Handle nearly new Moon.
    # Geometrically this is almost zero illuminated area.
    if np.pi - alpha_rad < 1.0e-8:
        return np.empty((0, 2))

    v = np.linspace(-1.0, 1.0, n)

    sqrt_term = np.sqrt(np.clip(1.0 - v**2, 0.0, None))

    # Bright circular limb.
    u_limb = sqrt_term

    # Projected terminator.
    u_term = -np.cos(alpha_rad) * sqrt_term

    # Boundary path:
    # 1. Go up along the bright circular limb.
    # 2. Return down along the terminator.
    u_path = np.concatenate([u_limb, u_term[::-1]])
    v_path = np.concatenate([v, v[::-1]])

    # Convert from (u, v) coordinates into plot/image (x, y) coordinates.
    vertices_xy = u_path[:, None] * bright_dir[None, :] + v_path[:, None] * perp_dir[None, :]

    return vertices_xy


if __name__ == "__main__":
    result = plot_moon_phase_from_location(when="2026-07-24T04:00:00Z", ephemeris_filename="de430t.bsp")
    print("done")
