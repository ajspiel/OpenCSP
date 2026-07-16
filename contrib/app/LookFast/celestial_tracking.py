import csv
from pathlib import Path
from logging import DEBUG, ERROR
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import skyfield.api as skf
import numpy as np

import opencsp.common.lib.tool.file_tools as ft

# import opencsp.app.lookback.lookback_tools as lbt
import contrib.app.LookFast.lookback_tools as lbt


def dms_to_decimal_degrees(dms):
    """
    Convert degrees/minutes/seconds to decimal degrees.

    Parameters
    ----------
    dms : tuple
        Tuple of the form:

            (degrees, minutes, seconds)

        The sign should generally be placed on the degrees component.

        Examples:
            (39, 45, 32.6232) -> positive latitude
            (-104, 37, 12.4206) -> west longitude

    Returns
    -------
    float
        Decimal degrees.
    """
    if not isinstance(dms, (tuple, list)) or len(dms) != 3:
        raise ValueError("DMS value must be a tuple/list like (degrees, minutes, seconds).")

    degrees, minutes, seconds = dms

    degrees = float(degrees)
    minutes = float(minutes)
    seconds = float(seconds)

    if minutes < 0 or seconds < 0:
        raise ValueError("Minutes and seconds should be non-negative. " "Put the sign on the degrees component.")

    if minutes >= 60 or seconds >= 60:
        raise ValueError("Minutes and seconds should each be less than 60.")

    sign = -1.0 if degrees < 0 else 1.0

    decimal_degrees = sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)

    return decimal_degrees


def parse_observer_location_dms(observer_location):
    """
    Parse an observer location of the form:

        (
            (latitude_degrees, latitude_minutes, latitude_seconds),
            (longitude_degrees, longitude_minutes, longitude_seconds),
            elevation_meters
        )

    Returns
    -------
    tuple
        (latitude_decimal_degrees, longitude_decimal_degrees, elevation_meters)
    """
    if not isinstance(observer_location, (tuple, list)) or len(observer_location) != 3:
        raise ValueError(
            "observer_location must be of the form "
            "((lat_deg, lat_min, lat_sec), (lon_deg, lon_min, lon_sec), elevation_m)."
        )

    latitude_dms, longitude_dms, elevation_m = observer_location

    latitude_deg = dms_to_decimal_degrees(latitude_dms)
    longitude_deg = dms_to_decimal_degrees(longitude_dms)
    elevation_m = float(elevation_m)

    return latitude_deg, longitude_deg, elevation_m


def _to_utc_datetime(value):
    """
    Convert a supported time input to a timezone-aware UTC datetime.

    Supported inputs:
      - datetime.datetime
      - ISO-8601 string, e.g. "2026-07-28T23:00:00-06:00"
      - ISO-8601 string with Z, e.g. "2026-07-29T05:00:00Z"
      - tuple/list: (year, month, day, hour, minute, second)

    If a datetime or tuple is naive, UTC is assumed.
    """
    if isinstance(value, datetime):
        dt = value

    elif isinstance(value, str):
        text = value.strip()

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        dt = datetime.fromisoformat(text)

    elif isinstance(value, (tuple, list)):
        dt = datetime(*value)

    else:
        raise TypeError(
            "Time values must be datetime objects, ISO-8601 strings, "
            "or tuples like (year, month, day, hour, minute, second)."
        )

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def generate_evenly_spaced_datetimes(start_time, stop_time, number_of_observations):
    """
    Generate evenly spaced UTC datetimes from start_time to stop_time.

    If number_of_observations == 1, only start_time is returned.

    If number_of_observations > 1, the returned list includes both
    start_time and stop_time.
    """
    if number_of_observations < 1:
        raise ValueError("number_of_observations must be at least 1.")

    start_dt = _to_utc_datetime(start_time)
    stop_dt = _to_utc_datetime(stop_time)

    if number_of_observations == 1:
        return [start_dt]

    if stop_dt < start_dt:
        raise ValueError("stop_time must be greater than or equal to start_time.")

    total_seconds = (stop_dt - start_dt).total_seconds()

    observation_datetimes = [
        start_dt + timedelta(seconds=total_seconds * i / (number_of_observations - 1))
        for i in range(number_of_observations)
    ]

    return observation_datetimes


def write_celestial_az_el_vectors_csv(
    observer_location,
    start_time,
    stop_time,
    number_of_observations,
    celestial_body_name,
    output_csv_path,
    ephemeris_file_path=None,
    ephemeris_filename="de430t.bsp",
):
    """
    Write a CSV file containing azimuth, elevation, range, and local unit vectors
    from a ground observer to a celestial body.

    Parameters
    ----------
    observer_location : tuple
        Observer location as:

            (
                (latitude_degrees, latitude_minutes, latitude_seconds),
                (longitude_degrees, longitude_minutes, longitude_seconds),
                elevation_meters
            )

        Example:

            ((39, 45, 32.6232), (-104, 37, 12.4206), 1672.4)

        For west longitude, use a negative degree component.

    start_time : datetime | str | tuple
        Start time.

        Timezone-aware datetimes are recommended.

    stop_time : datetime | str | tuple
        Stop time.

    number_of_observations : int
        Number of observation times to generate.

        If greater than 1, observations are evenly spaced between
        start_time and stop_time, including both endpoints.

    celestial_body_name : str
        Name of the celestial body in the Skyfield ephemeris.

        Common examples:

            "sun"
            "moon"

    output_csv_path : str | pathlib.Path
        Path where the CSV file should be written.

    ephemeris_file_path : str | pathlib.Path | None
        Optional path to the ephemeris file or to a directory containing it.

    ephemeris_filename : str
        Ephemeris filename to use when ephemeris_file_path is a directory.

    Returns
    -------
    pathlib.Path
        Path to the written CSV file.

    CSV Columns
    -----------
    index
        Observation index.

    utc_iso
        UTC observation time.

    celestial_body
        Name of the observed celestial body.

    observer_latitude_deg
        Observer latitude in decimal degrees.

    observer_longitude_deg
        Observer longitude in decimal degrees.

    observer_elevation_m
        Observer elevation in meters.

    azimuth_deg
        Azimuth in degrees, measured clockwise from north.

    elevation_deg
        Elevation angle in degrees above the local horizon.

    range_km
        Apparent range from observer to celestial body in kilometers.

    unit_east
        East component of the local ENU unit vector.

    unit_north
        North component of the local ENU unit vector.

    unit_up
        Up/zenith component of the local ENU unit vector.
    """
    observer_latitude_deg, observer_longitude_deg, observer_elevation_m = parse_observer_location_dms(observer_location)

    observation_datetimes = generate_evenly_spaced_datetimes(
        start_time=start_time, stop_time=stop_time, number_of_observations=number_of_observations
    )

    output_csv_path = Path(output_csv_path).expanduser().resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    eph = lbt.load_de430_ephemeris(ephemeris_path=ephemeris_file_path, filename=ephemeris_filename)

    body_key = celestial_body_name.lower().strip()

    try:
        celestial_body = eph[body_key]
    except KeyError as exc:
        raise KeyError(
            f"Celestial body '{celestial_body_name}' was not found in the ephemeris. "
            f"Try names like 'sun' or 'moon'."
        ) from exc

    earth = eph["earth"]

    observer_topos = skf.wgs84.latlon(
        latitude_degrees=observer_latitude_deg,
        longitude_degrees=observer_longitude_deg,
        elevation_m=observer_elevation_m,
    )

    observer = earth + observer_topos

    ts = skf.load.timescale()
    t = ts.from_datetimes(observation_datetimes)

    apparent = observer.at(t).observe(celestial_body).apparent()

    altitude, azimuth, distance = apparent.altaz()

    azimuth_deg = np.atleast_1d(azimuth.degrees)
    elevation_deg = np.atleast_1d(altitude.degrees)
    range_km = np.atleast_1d(distance.km)

    # Convert azimuth/elevation to local ENU unit vector.
    #
    # Skyfield azimuth convention:
    #   0 deg   = north
    #   90 deg  = east
    #   180 deg = south
    #   270 deg = west
    #
    # Local vector convention here:
    #   x = east
    #   y = north
    #   z = up
    azimuth_rad = np.radians(azimuth_deg)
    elevation_rad = np.radians(elevation_deg)

    unit_east = np.cos(elevation_rad) * np.sin(azimuth_rad)
    unit_north = np.cos(elevation_rad) * np.cos(azimuth_rad)
    unit_up = np.sin(elevation_rad)

    with output_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "index",
                "utc_iso",
                "celestial_body",
                "observer_latitude_deg",
                "observer_longitude_deg",
                "observer_elevation_m",
                "azimuth_deg",
                "elevation_deg",
                "range_km",
                "unit_east",
                "unit_north",
                "unit_up",
            ]
        )

        for i, dt in enumerate(observation_datetimes):
            writer.writerow(
                [
                    i,
                    dt.isoformat().replace("+00:00", "Z"),
                    body_key,
                    float(observer_latitude_deg),
                    float(observer_longitude_deg),
                    float(observer_elevation_m),
                    float(azimuth_deg[i]),
                    float(elevation_deg[i]),
                    float(range_km[i]),
                    float(unit_east[i]),
                    float(unit_north[i]),
                    float(unit_up[i]),
                ]
            )

    return output_csv_path


if __name__ == "__main__":
    observer_location = (
        (39, 45, 32.6232),  # latitude:  39 deg 45 min 32.6232 sec N
        (-104, 37, 12.4206),  # longitude: 104 deg 37 min 12.4206 sec W
        1672.4,  # elevation in meters
    )

    denver_tz = ZoneInfo("America/Denver")

    start_time = datetime(2026, 7, 27, 12, 0, 0, tzinfo=denver_tz)

    stop_time = datetime(2026, 7, 31, 12, 0, 0, tzinfo=denver_tz)

    csv_path = write_celestial_az_el_vectors_csv(
        observer_location=observer_location,
        start_time=start_time,
        stop_time=stop_time,
        number_of_observations=3840,
        celestial_body_name="sun",
        output_csv_path="sun_az_el_vectors_denver.csv",
        ephemeris_file_path=None,
    )
