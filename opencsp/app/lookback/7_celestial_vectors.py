import re
import os
import json
import gzip
import time
import subprocess
from multiprocessing import Pool, Manager
import logging

# import warnings
from zoneinfo import ZoneInfo
from datetime import datetime, timezone, timedelta
import skyfield.api as skf
import numpy as np
from tqdm import tqdm

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Specify the folder where the log file should be saved
log_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/0_checkpoints/error_logs"  # Replace with your desired folder path
log_file = os.path.join(log_folder, "error_log.txt")

# Ensure the folder exists
os.makedirs(log_folder, exist_ok=True)

# Create a custom logger
logger = logging.getLogger(__name__)

# Set the logging level
logger.setLevel(logging.ERROR)

# Create a file handler
file_handler = logging.FileHandler(log_file)

# Set the level for the file handler
file_handler.setLevel(logging.ERROR)

# Create a formatter and add it to the file handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


def save_checkpoint(checkpoint_folder, checkpoint_file_name, checkpoint_data, print_path=True):
    """
    Saves the checkpoint data to a JSON file.

    Parameters:
        checkpoint_file_name (str): Path to the checkpoint file.
        checkpoint_data (dict): Dictionary containing checkpoint information.

    Returns:
        None
    """
    raw_path = os.path.join(checkpoint_folder, checkpoint_file_name)
    norm_path = os.path.normpath(raw_path)

    success = False
    for i in range(10):
        try:
            with open(norm_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=4)
                success = True
                break
        except Exception:
            logger.error("Unable to save checkpoint", exc_info=True)
            time.sleep(0.01)

    if success:
        if print_path and i == 0:
            logger.info("Checkpoint saved to %s", raw_path)
        if i > 0:
            logger.info("Checkpoint saved to %s after %d attempts", raw_path, i)


def load_checkpoint(checkpoint_folder, checkpoint_file_name):
    """
    Loads the checkpoint data from a JSON file.

    Parameters:
        checkpoint_folder (str): Path to the folder containing the checkpoint file.
        checkpoint_file_name (str): Name of the checkpoint file.

    Returns:
        dict: Dictionary containing checkpoint information, or None if the file is empty or does not exist.
    """
    norm_path = os.path.normpath(os.path.join(checkpoint_folder, checkpoint_file_name))

    if os.path.exists(norm_path):
        # Check if the file is empty
        if os.path.getsize(norm_path) == 0:
            print(f"Checkpoint file {norm_path} is empty.")
            return None

        # Attempt to load the JSON data
        with open(norm_path, 'r', encoding='utf-8') as f:
            try:
                checkpoint_data = json.load(f)
                # Check if the loaded JSON is empty (e.g., {} or [])
                if not checkpoint_data:
                    print(f"Checkpoint file {norm_path} contains empty JSON data.")
                    return None
                print(f"Checkpoint loaded from {norm_path}")
                return checkpoint_data
            except json.JSONDecodeError:
                logger.error("Invalid Json File", exc_info=True)
                print(f"Checkpoint file {norm_path} contains invalid JSON.")
                return None
    else:
        print(f"Checkpoint file {norm_path} does not exist.")
        return None


def custom_serializer(obj):
    """
    Custom serializer to handle numpy arrays, datetime objects, floats, integers, and strings.
    Converts unsupported types into JSON-compatible formats.
    """
    try:
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list

        # Handle datetime objects
        elif isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to ISO 8601 string

        # Handle floats
        elif isinstance(obj, float):
            return float(obj)  # Ensure it's a float (JSON-compatible)

        # Handle integers
        elif isinstance(obj, int):
            return int(obj)  # Ensure it's an integer (JSON-compatible)

        # Handle strings
        elif isinstance(obj, str):
            return str(obj)  # Ensure it's a string (JSON-compatible)

        # Handle sets (convert to list for JSON compatibility)
        elif isinstance(obj, set):
            return list(obj)  # Convert set to list

        # Unsupported type
        else:
            logger.error("Unsupported type encountered: %r", type(obj), exc_info=True)
            raise TypeError(f"Type {type(obj)} not serializable")

    except Exception:
        logger.error("Error during serialization", exc_info=True)
        raise


def custom_deserializer(obj):
    """
    Custom deserializer to handle strings, datetime strings, numpy arrays, floats, and integers.
    Converts JSON-compatible formats back into their original types.
    """
    for key, value in obj.items():
        # Handle numpy arrays (lists of numbers)
        if isinstance(value, list) and all(isinstance(i, (int, float)) for i in value):
            try:
                obj[key] = np.array(value)  # Convert list back to numpy array
            except ValueError:
                logger.error("Failed to convert %s to numpy array", key, exc_info=True)
                # If conversion fails, leave as list

        # Handle datetime strings
        elif isinstance(value, str):
            try:
                obj[key] = datetime.fromisoformat(value)  # Convert ISO 8601 string back to datetime
            except ValueError:
                logger.debug("%s is not a valid datetime string, leaving as string.", key, exc_info=True)
                # If conversion fails, leave as string

        # Handle floats explicitly
        elif isinstance(value, float):
            obj[key] = float(value)  # Ensure it's a float (redundant but explicit)

        # Handle integers explicitly
        elif isinstance(value, int):
            obj[key] = int(value)  # Ensure it's an integer (redundant but explicit)

        # Handle strings explicitly (fallback case)
        elif isinstance(value, str):
            obj[key] = str(value)  # Ensure it's a string (redundant but explicit)

    return obj


def read_json(file_path):
    """
    Reads a regular JSON file and returns the data.

    Parameters:
        file_path (str): Path to the .json file.

    Returns:
        dict or list: Data from the file.
    """
    norm_path = os.path.normpath(file_path)
    try:
        with open(norm_path, 'r', encoding='utf-8') as f:
            data = json.load(f, object_hook=custom_deserializer)
        return data
    except Exception:
        logger.error("read_json Error", exc_info=True)
        return None


def write_json(data, file_path):
    """
    Writes data to a regular JSON file.

    Parameters:
        data (dict or list): Data to write to the file.
        file_path (str): Path to the .json file.

    Returns:
        None
    """
    norm_path = os.path.normpath(file_path)
    try:
        with open(norm_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, default=custom_serializer)
        print(f"Data written to {norm_path}")
    except Exception:
        logger.error("write_json Error", exc_info=True)


def read_compressed_json(file_path):
    """
    Reads a compressed JSON file (.json.gz) and returns the data.

    Parameters:
        file_path (str): Path to the .json.gz file.

    Returns:
        list: List of dictionaries containing the data from the file.
    """
    norm_path = os.path.normpath(file_path)
    try:
        with gzip.open(norm_path, 'rt', encoding='utf-8') as f:
            data = json.load(f, object_hook=custom_deserializer)
        return data
    except Exception:
        logger.error("read_compressed_json Error", exc_info=True)
        return None


def write_compressed_json(data, file_path, print_path=True):
    """
    Writes data to a compressed JSON file (.json.gz).

    Parameters:
        data (dict or list): Data to write to the file.
        file_path (str): Path to the .json.gz file.

    Returns:
        None
    """
    norm_path = os.path.normpath(file_path)
    success = False
    for i in range(10):
        try:
            with gzip.open(norm_path, 'wt', encoding='utf-8') as f:
                try:
                    json.dump(data, f, indent=4, default=custom_serializer)
                    success = True
                    break
                except Exception:
                    logger.error("write_compressed_json Error", exc_info=True)
                    time.sleep(0.01)
        except Exception:
            logger.error("write_compressed_json Error", exc_info=True)

    if success:
        if print_path and i == 0:
            logger.info("File Saved to %s", norm_path)
        if i > 0:
            logger.info("File saved to %s after %d attempts", norm_path, i)


def extract_video_metadata_exiftool(video_path):
    """
    Extracts the media creation date, frame rate, and duration from a video file's metadata using exiftool.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        dict: A dictionary containing:
            - 'creation_date': Media creation date and time (if available).
            - 'frame_rate': Frame rate of the video (frames per second, if available).
            - 'duration': Duration of the video (seconds, if available).
    """
    try:
        # Run exiftool to extract metadata
        result = subprocess.run(
            [
                "exiftool",
                "-CreateDate",
                "-MediaCreateDate",
                "-DateTimeOriginal",
                "-VideoFrameRate",
                "-Duration",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        # Initialize metadata dictionary
        metadata = {'creation_date': None, 'frame_rate': None, 'duration': None}

        # Parse the output to find relevant metadata
        for line in result.stdout.splitlines():
            if "Create Date" in line or "Media Create Date" in line or "Date/Time Original" in line:
                metadata['creation_date'] = line.split(": ", 1)[1].strip()
            elif "Video Frame Rate" in line:
                metadata['frame_rate'] = float(
                    line.split(": ", 1)[1].strip().split(" ")[0]
                )  # Extract frame rate as float
            elif "Duration" in line:
                duration_str = line.split(": ", 1)[1].strip()
                # Convert duration to seconds (e.g., "0:01:23.456" -> 83.456 seconds)
                parts = duration_str.split(":")
                if len(parts) == 3:  # Format is hours:minutes:seconds
                    hours, minutes, seconds = map(float, parts)
                    metadata['duration'] = hours * 3600 + minutes * 60 + seconds
                elif len(parts) == 2:  # Format is minutes:seconds
                    minutes, seconds = map(float, parts)
                    metadata['duration'] = minutes * 60 + seconds

        return metadata
    except Exception:
        logger.error("extract_video_metadata_exiftool Error", exc_info=True)
        return None


def calculate_vectors_celestial_only(celestial_object_name, target_location, observation_time):
    """
    Calculates vectors from a celestial object to a target point and from the target point to an observer.

    Parameters:
        celestial_object_name (str): Name of the celestial object (e.g., "moon", "sun").
        target_location (tuple): Latitude, longitude, and elevation of the target point (degrees, meters).
        observation_time (tuple): Observation time as (year, month, day, hour, minute, second).

    Returns:
        dict: A dictionary containing the vectors:
            - "celestial_to_target": Vector from the celestial object to the target point (normalized).
            - "target_to_observer": Vector from the target point to the observer (normalized).
    """
    # Load ephemeris data (DE430 dataset)
    eph = skf.load("de430t.bsp")  # DE430 ephemeris file

    # Define celestial object
    celestial_object = eph[celestial_object_name]

    # Define target location
    target = skf.Topos(
        latitude_degrees=target_location[0], longitude_degrees=target_location[1], elevation_m=target_location[2]
    )

    # Define observation time
    ts = skf.load.timescale()
    if isinstance(observation_time, skf.Time):
        pass
    elif isinstance(observation_time, datetime):
        # Extract components from the datetime object
        observation_time = ts.utc(
            observation_time.year,
            observation_time.month,
            observation_time.day,
            observation_time.hour,
            observation_time.minute,
            observation_time.second,
        )
    else:
        # If observation_time is already a tuple, unpack it directly
        observation_time = ts.utc(*observation_time)

    # Calculate position of the celestial object relative to the target
    earth = eph["earth"]
    target_position_vec = earth + target
    target_position_bary = target_position_vec.at(observation_time)
    target_obsv_celest = target_position_bary.observe(celestial_object)
    target_obsv_celest_apparent = target_obsv_celest.apparent()

    celestial_to_target = target_position_vec.at(observation_time).observe(celestial_object).position.km
    celestial_to_target_normalized = celestial_to_target / np.linalg.norm(celestial_to_target)

    earth_target_bary = earth.at(observation_time)
    earth_obsv_target = earth_target_bary.observe(target_position_vec)
    earth_obsv_target_apparent = earth_obsv_target.apparent()

    NSTTF_coord_toca = [
        target_obsv_celest_apparent.frame_xyz(target).km[1],
        target_obsv_celest_apparent.frame_xyz(target).km[0],
        target_obsv_celest_apparent.frame_xyz(target).km[2],
    ]
    NSTTF_coord_eota = [
        earth_obsv_target_apparent.frame_xyz(target).km[1],
        earth_obsv_target_apparent.frame_xyz(target).km[0],
        earth_obsv_target_apparent.frame_xyz(target).km[2],
    ]
    # Calculate angular size in radians
    distance_to_celestial_object = target_obsv_celest_apparent.distance().km  # Distance in kilometers

    if celestial_object_name.lower() == 'moon':
        radius_of_celestial_object = 1737.4  # Radius in kilometers of the Moon
    elif celestial_object_name.lower() == 'sun':
        radius_of_celestial_object = 696340  # Radius in kilometers of the Sun
    else:
        radius_of_celestial_object = 1737.4  # Radius in kilometers of the Moon

    angular_size_radians = 2 * np.arctan(radius_of_celestial_object / distance_to_celestial_object)

    # Apparent function returns a tuple with (Altitude, Azimuth, and Distance)
    # Altitude measures the angle above or below the horizon. The zenith is at +90°, an object on the horizon’s great circle is at 0°, and the nadir beneath your feet is at −90°.
    # Azimuth measures the angle around the sky from the north pole: 0° means exactly north, 90° is east, 180° is south, and 270° is west.
    # "earth_to_target_spherical": earth_obsv_target_apparent.altaz()
    # Cartesian for horizonal skyfield coordinates is Left Handed {x points north, y points east, z points to zenith}
    # NSTTF coordinates is Right Handed {x points east, y points north, z points to zenith}
    return {
        "celestial_to_target": celestial_to_target_normalized,
        "cel_to_target_cartesian": NSTTF_coord_toca / np.linalg.norm(NSTTF_coord_toca),
        "earth_to_target_cartesian": NSTTF_coord_eota / np.linalg.norm(NSTTF_coord_eota),
        "angular_size_radians": angular_size_radians,
    }


def calculate_vectors_celestial_observer(celestial_object_name, target_location, observer_location, observation_time):
    """
    Calculates vectors from a celestial object to a target point and from the target point to an observer.

    Parameters:
        celestial_object_name (str): Name of the celestial object (e.g., "moon", "sun").
        target_location (tuple): Latitude, longitude, and elevation of the target point (degrees, meters).
        observation_time (tuple): Observation time as (year, month, day, hour, minute, second).

    Returns:
        dict: A dictionary containing the vectors:
            - "celestial_to_target": Vector from the celestial object to the target point (normalized).
            - "target_to_observer": Vector from the target point to the observer (normalized).
    """
    # Load ephemeris data (DE430 dataset)
    eph = skf.load("de430t.bsp")  # DE430 ephemeris file

    # Define celestial object
    celestial_object = eph[celestial_object_name]

    # Define target location
    target = skf.Topos(
        latitude_degrees=target_location[0], longitude_degrees=target_location[1], elevation_m=target_location[2]
    )
    # Define observer location
    observer = skf.Topos(
        latitude_degrees=observer_location[0], longitude_degrees=observer_location[1], elevation_m=observer_location[2]
    )

    # Define observation time
    ts = skf.load.timescale()
    if isinstance(observation_time, skf.Time):
        pass
    elif isinstance(observation_time, datetime):
        # Extract components from the datetime object
        observation_time = ts.utc(
            observation_time.year,
            observation_time.month,
            observation_time.day,
            observation_time.hour,
            observation_time.minute,
            observation_time.second,
        )
    else:
        # If observation_time is already a tuple, unpack it directly
        observation_time = ts.utc(*observation_time)

    # Calculate position of the celestial object relative to the target
    earth = eph["earth"]
    target_position_vec = earth + target
    target_position_bary = target_position_vec.at(observation_time)
    target_obsv_celest = target_position_bary.observe(celestial_object)
    target_obsv_celest_apparent = target_obsv_celest.apparent()

    celestial_to_target = target_position_vec.at(observation_time).observe(celestial_object).position.km
    celestial_to_target_normalized = celestial_to_target / np.linalg.norm(celestial_to_target)

    # Calculate position of the target relative to the observer
    observer_position = earth + observer
    target_to_observer = observer_position.at(observation_time).observe(target_position_vec)

    earth_target_bary = earth.at(observation_time)
    earth_obsv_target = earth_target_bary.observe(target_position_vec)
    earth_obsv_target_apparent = earth_obsv_target.apparent()

    NSTTF_coord_toca = [
        target_obsv_celest_apparent.frame_xyz(target).km[1],
        target_obsv_celest_apparent.frame_xyz(target).km[0],
        target_obsv_celest_apparent.frame_xyz(target).km[2],
    ]
    NSTTF_coord_eota = [
        earth_obsv_target_apparent.frame_xyz(target).km[1],
        earth_obsv_target_apparent.frame_xyz(target).km[0],
        earth_obsv_target_apparent.frame_xyz(target).km[2],
    ]
    NSTTF_coord_otta = [
        target_to_observer.frame_xyz(target).km[1],
        target_to_observer.frame_xyz(target).km[0] * -1,
        target_to_observer.frame_xyz(target).km[2] * -1,
    ]
    # Calculate angular size in radians
    distance_to_celestial_object = target_obsv_celest_apparent.distance().km  # Distance in kilometers

    if celestial_object_name.lower() == 'moon':
        radius_of_celestial_object = 1737.4  # Radius in kilometers of the Moon
    elif celestial_object_name.lower() == 'sun':
        radius_of_celestial_object = 696340  # Radius in kilometers of the Sun
    else:
        radius_of_celestial_object = 1737.4  # Radius in kilometers of the Moon

    angular_size_radians = 2 * np.arctan(radius_of_celestial_object / distance_to_celestial_object)

    # Apparent function returns a tuple with (Altitude, Azimuth, and Distance)
    # Altitude measures the angle above or below the horizon. The zenith is at +90°, an object on the horizon’s great circle is at 0°, and the nadir beneath your feet is at −90°.
    # Azimuth measures the angle around the sky from the north pole: 0° means exactly north, 90° is east, 180° is south, and 270° is west.
    # "earth_to_target_spherical": earth_obsv_target_apparent.altaz()
    # Cartesian for horizonal skyfield coordinates is Left Handed {x points north, y points east, z points to zenith}
    # NSTTF coordinates is Right Handed {x points east, y points north, z points to zenith}
    return {
        "celestial_to_target": celestial_to_target_normalized,
        "cel_to_target_cartesian": NSTTF_coord_toca / np.linalg.norm(NSTTF_coord_toca),
        "earth_to_target_cartesian": NSTTF_coord_eota / np.linalg.norm(NSTTF_coord_eota),
        "target_to_observer": NSTTF_coord_otta / np.linalg.norm(NSTTF_coord_otta),
        "angular_size_radians": angular_size_radians,
    }


def extract_pixel_timing_and_celestial_vectors(
    celestial_object_name,
    target_location,
    observer_location,
    camera_time_shift,
    data_location,
    video_metadata,
    output_folder_vector,
    output_json_name,
    checkpoint_folder,
    checkpoint_file,
):
    video_start_time = datetime.strptime(video_metadata['creation_date'], "%Y:%m:%d %H:%M:%S")
    abq_tz = ZoneInfo("America/Denver")
    video_start_time = video_start_time.replace(tzinfo=abq_tz)
    # Read in pixel timing information
    # Load checkpoint if it exists
    checkpoint_data = load_checkpoint(checkpoint_folder, checkpoint_file)
    if checkpoint_data is None:
        checkpoint_data = {"processed_pixels": [], "pixels_with_data": [], "pixels_without_data": []}
    data = read_compressed_json(data_location)
    # Create a tqdm progress bar outside the loop
    progress_bar = tqdm(total=len(data), desc="Calculating Pixel Vector Information")

    for pixel, transitions in data.items():
        if pixel in checkpoint_data["processed_pixels"]:
            # print(f"Skipping already processed pixel {pixel}.")
            progress_bar.update(1)
            continue

        frame_diff = 0
        if len(data[pixel]) == 0:
            checkpoint_data["processed_pixels"].append(pixel)
            checkpoint_data["pixels_without_data"].append(pixel)
            save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
            progress_bar.update(1)
            continue
        elif len(data[pixel]) < 2:
            checkpoint_data["processed_pixels"].append(pixel)
            checkpoint_data["pixels_without_data"].append(pixel)
            save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
            progress_bar.update(1)
            continue
        elif len(data[pixel]) == 2:
            frame_range = []
            for transition in transitions:
                if transition["transition"] == "bright":
                    frame_range.append(tuple((1, frame_number_from_img_name(transition['to_frame']))))
                elif transition["transition"] == "dark":
                    frame_range.append(tuple((0, frame_number_from_img_name(transition['to_frame']))))
            frame_diff = frame_range[1][1] - frame_range[0][1]
        elif len(data[pixel]) > 2:
            frame_range_all = []
            for transition in transitions:
                if transition["transition"] == "bright":
                    frame_range_all.append(tuple((1, frame_number_from_img_name(transition['to_frame']))))
                elif transition["transition"] == "dark":
                    frame_range_all.append(tuple((0, frame_number_from_img_name(transition['to_frame']))))
            frame_range, frame_diff = maximum_frame_range(frame_range=frame_range_all)

        elapsed_time_bright = frame_diff / video_metadata['frame_rate']
        elapsed_time_start = frame_range[0][1] / video_metadata['frame_rate']

        bright_start_time = video_start_time + timedelta(seconds=elapsed_time_start)
        bright_end_time = (
            video_start_time + timedelta(seconds=elapsed_time_start) + timedelta(seconds=elapsed_time_bright)
        )

        obsv_time_utc_start = define_observation_time(bright_start_time, camera_time_shift)
        obsv_time_utc_end = define_observation_time(bright_end_time, camera_time_shift)

        start_vector = calculate_vectors_celestial_observer(
            celestial_object_name=celestial_object_name,
            target_location=target_location,
            observer_location=observer_location,
            observation_time=obsv_time_utc_start,
        )
        end_vector = calculate_vectors_celestial_observer(
            celestial_object_name=celestial_object_name,
            target_location=target_location,
            observer_location=observer_location,
            observation_time=obsv_time_utc_end,
        )
        points = np.array([[0, 0, 0], start_vector['cel_to_target_cartesian'], end_vector['cel_to_target_cartesian']])
        radii = np.array([1, start_vector['angular_size_radians'] / 2, end_vector['angular_size_radians'] / 2])
        try:
            intersection_1, intersection_2 = trilaterate(points, radii, raise_on_no_solution=True)
        except ValueError:
            logger.debug("extract_pixel_timing_and_celestial_vectors Error pixel %s", pixel, exc_info=True)
            checkpoint_data["processed_pixels"].append(pixel)
            checkpoint_data["pixels_without_data"].append(pixel)
            save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
            progress_bar.update(1)
            continue

        observer_vec, slope_1, slope_2 = calculate_slope(
            start_vector['target_to_observer'], end_vector['target_to_observer'], intersection_1, intersection_2
        )

        data[pixel].append(
            {
                "start_time_Local": obsv_time_utc_start.astimezone(abq_tz),
                "start_time_UTC": obsv_time_utc_start.utc_datetime(),
                "start_vector": start_vector,
                "end_time_Local": obsv_time_utc_end.astimezone(abq_tz),
                "end_time_UTC": obsv_time_utc_end.utc_datetime(),
                "end_vector": end_vector,
                "intersection_1": intersection_1,
                "intersection_2": intersection_2,
                "slope_1": slope_1,
                "slope_2": slope_2,
                "observer_vector": observer_vec,
            }
        )
        checkpoint_data["processed_pixels"].append(pixel)
        checkpoint_data["pixels_with_data"].append(pixel)
        save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
        write_compressed_json(
            data=data, file_path=os.path.join(output_folder_vector, output_json_name), print_path=False
        )
        # write_json(data=data, file_path=os.path.join(output_folder_vector, output_json_name[:-3]))
        progress_bar.update(1)
    progress_bar.close()


def process_pixel(args):
    (
        pixel,
        transitions,
        video_metadata,
        video_start_time,
        camera_time_shift,
        celestial_object_name,
        target_location,
        observer_location,
        abq_tz,
    ) = args

    checkpoint_data = {"processed_pixels": [], "pixels_with_data": [], "pixels_without_data": []}

    frame_diff = 0
    if len(transitions) == 0 or len(transitions) < 2:
        checkpoint_data["processed_pixels"].append(pixel)
        checkpoint_data["pixels_without_data"].append(pixel)
        # save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
        return pixel, None, checkpoint_data

    if len(transitions) == 2:
        frame_range = []
        for transition in transitions:
            if transition["transition"] == "bright":
                frame_range.append(tuple((1, frame_number_from_img_name(transition['to_frame']))))
            elif transition["transition"] == "dark":
                frame_range.append(tuple((0, frame_number_from_img_name(transition['to_frame']))))
        frame_diff = frame_range[1][1] - frame_range[0][1]
    elif len(transitions) > 2:
        frame_range_all = []
        for transition in transitions:
            if transition["transition"] == "bright":
                frame_range_all.append(tuple((1, frame_number_from_img_name(transition['to_frame']))))
            elif transition["transition"] == "dark":
                frame_range_all.append(tuple((0, frame_number_from_img_name(transition['to_frame']))))
        frame_range, frame_diff = maximum_frame_range(frame_range=frame_range_all)

    elapsed_time_bright = frame_diff / video_metadata['frame_rate']
    elapsed_time_start = frame_range[0][1] / video_metadata['frame_rate']

    bright_start_time = video_start_time + timedelta(seconds=elapsed_time_start)
    bright_end_time = video_start_time + timedelta(seconds=elapsed_time_start) + timedelta(seconds=elapsed_time_bright)

    obsv_time_utc_start = define_observation_time(bright_start_time, camera_time_shift)
    obsv_time_utc_end = define_observation_time(bright_end_time, camera_time_shift)

    start_vector = calculate_vectors_celestial_observer(
        celestial_object_name=celestial_object_name,
        target_location=target_location,
        observer_location=observer_location,
        observation_time=obsv_time_utc_start,
    )
    end_vector = calculate_vectors_celestial_observer(
        celestial_object_name=celestial_object_name,
        target_location=target_location,
        observer_location=observer_location,
        observation_time=obsv_time_utc_end,
    )
    points = np.array([[0, 0, 0], start_vector['cel_to_target_cartesian'], end_vector['cel_to_target_cartesian']])
    radii = np.array([1, start_vector['angular_size_radians'] / 2, end_vector['angular_size_radians'] / 2])
    try:
        intersection_1, intersection_2 = trilaterate(points, radii, raise_on_no_solution=True)
    except ValueError:
        logger.debug("process_pixel Error pixel %s", pixel, exc_info=True)
        checkpoint_data["processed_pixels"].append(pixel)
        checkpoint_data["pixels_without_data"].append(pixel)
        # save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
        return pixel, None, checkpoint_data

    observer_vec, slope_1, slope_2 = calculate_slope(
        start_vector['target_to_observer'], end_vector['target_to_observer'], intersection_1, intersection_2
    )

    pixel_data = {
        "start_time_Local": obsv_time_utc_start.astimezone(abq_tz),
        "start_time_UTC": obsv_time_utc_start.utc_datetime(),
        "start_vector": start_vector,
        "end_time_Local": obsv_time_utc_end.astimezone(abq_tz),
        "end_time_UTC": obsv_time_utc_end.utc_datetime(),
        "end_vector": end_vector,
        "intersection_1": intersection_1,
        "intersection_2": intersection_2,
        "slope_1": slope_1,
        "slope_2": slope_2,
        "observer_vector": observer_vec,
    }

    checkpoint_data["processed_pixels"].append(pixel)
    checkpoint_data["pixels_with_data"].append(pixel)
    # save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)

    return pixel, pixel_data, checkpoint_data


def extract_pixel_timing_and_celestial_vectors_parallel(
    celestial_object_name,
    target_location,
    observer_location,
    camera_time_shift,
    data_location,
    video_metadata,
    output_folder_vector,
    output_json_name,
    checkpoint_folder,
    checkpoint_file,
    batch_size=500,  # Number of pixels to process in each batch
):
    video_start_time = datetime.strptime(video_metadata['creation_date'], "%Y:%m:%d %H:%M:%S")
    abq_tz = ZoneInfo("America/Denver")
    video_start_time = video_start_time.replace(tzinfo=abq_tz)

    checkpoint_data = load_checkpoint(checkpoint_folder, checkpoint_file)
    if checkpoint_data is None:
        checkpoint_data = {"processed_pixels": [], "pixels_with_data": [], "pixels_without_data": []}

    # Ensure the output folder exists
    os.makedirs(output_folder_vector, exist_ok=True)

    # Read compressed JSON file in chunks
    with gzip.open(data_location, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    # Convert data to a list of items for batching
    data_items = list(data.items())
    total_pixels = len(data_items)

    # Process data in batches
    for batch_start in range(0, total_pixels, batch_size):
        batch_end = min(batch_start + batch_size, total_pixels)
        batch_data = data_items[batch_start:batch_end]
        # Filter batch_data to exclude already processed pixels
        filtered_batch_data = [
            (pixel, transitions)
            for pixel, transitions in batch_data
            if pixel not in checkpoint_data["processed_pixels"]
        ]
        # Use multiprocessing to parallelize the pixel processing
        with Manager() as manager:
            args_list = [
                (
                    pixel,
                    transitions,
                    video_metadata,
                    video_start_time,
                    camera_time_shift,
                    celestial_object_name,
                    target_location,
                    observer_location,
                    abq_tz,
                )
                for pixel, transitions in filtered_batch_data
            ]

            with Pool(processes=os.cpu_count()) as pool:  # Use all available CPU cores
                results = list(
                    tqdm(
                        pool.imap(process_pixel, args_list),
                        total=len(filtered_batch_data),
                        desc=f"Processing Batch {batch_start}-{batch_end}",
                    )
                )

            # Aggregate checkpoint data
            for _, _, pixel_checkpoint_data in results:
                ppixel = pixel_checkpoint_data["processed_pixels"]
                dpixel = pixel_checkpoint_data["pixels_with_data"]
                wpixel = pixel_checkpoint_data["pixels_without_data"]
                checkpoint_data["processed_pixels"].append(ppixel[0])
                if dpixel:
                    checkpoint_data["pixels_with_data"].append(dpixel[0])
                if wpixel:
                    checkpoint_data["pixels_without_data"].append(wpixel[0])

            # Save checkpoint data serially after each batch
            save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)

            # Update the data with processed results
            for pixel, pixel_data, _ in results:
                if pixel_data is not None:
                    data[pixel] = pixel_data

            # Save intermediate results to avoid data loss
            if results:
                write_compressed_json(
                    data=data,
                    file_path=os.path.join(
                        output_folder_vector, f"{output_json_name}_batch_{batch_start}_{batch_end}.json.gz"
                    ),
                    print_path=False,
                )

    # Save final results
    if os.path.exists(os.path.join(output_folder_vector, output_json_name)):
        pass
    else:
        write_compressed_json(
            data=data, file_path=os.path.join(output_folder_vector, output_json_name), print_path=True
        )


def calculate_slope(observer_vec_start, observer_vec_end, inter_1, inter_2):
    observer_vec = (observer_vec_start + observer_vec_end) / 2
    slope_1 = (observer_vec + inter_1) / np.linalg.norm(observer_vec + inter_1)
    slope_2 = (observer_vec + inter_2) / np.linalg.norm(observer_vec + inter_2)
    return observer_vec, slope_1, slope_2


def plotting_pixel_transition_vectors(pixel, vector_dict, celestial_object, output_folder):
    start_vector = vector_dict['start_vector']
    end_vector = vector_dict['end_vector']
    local_time_start = vector_dict['start_time_Local']
    local_time_end = vector_dict['end_time_Local']
    inter_1 = vector_dict['intersection_1']
    inter_2 = vector_dict['intersection_2']
    observer_vec = vector_dict['observer_vector']
    # plotting vectors
    # Create a the top half of a unit sphere for visualization of the horizon
    u = np.linspace(np.pi, 2 * np.pi, 100)
    v = np.linspace(0, np.pi / 2, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Initialize 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the unit sphere
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.15)  # Initialize 3D plot

    # Initialize inset 3D plot
    inset_ax = fig.add_axes([0.02, 0.05, 0.3, 0.3], projection='3d')  # Adjust position and size of inset plot

    # Plot the unit sphere on the inset plot
    inset_ax.plot_surface(x, y, z, color='lightblue', alpha=0.15)

    ins_x_lim = []
    ins_y_lim = []
    ins_z_lim = []
    # Plot the projections for each interpolated time
    for i, vector in enumerate([start_vector, end_vector]):

        # Convert angular size to radians and calculate angular radius
        angular_radius = vector['angular_size_radians'] / 2
        # Create a circle in 3D space for the projection
        num_points = 360  # Number of points to define the circle
        theta = np.linspace(0, 2 * np.pi, num_points)  # Angle around the circle

        # Find a basis for the plane perpendicular to the vector
        arbitrary_vector = (
            np.array([1, 0, 0])
            if not np.allclose(vector['cel_to_target_cartesian'], [1, 0, 0])
            else np.array([0, 1, 0])
        )
        basis1 = np.cross(vector['cel_to_target_cartesian'], arbitrary_vector)
        basis1 /= np.linalg.norm(basis1)  # Normalize

        basis2 = np.cross(vector['cel_to_target_cartesian'], basis1)
        basis2 /= np.linalg.norm(basis2)  # Normalize

        # Generate points on the circle
        circle_points = []
        for angle in theta:
            point = np.cos(angular_radius) * vector['cel_to_target_cartesian'] + np.sin(angular_radius) * (
                np.cos(angle) * basis1 + np.sin(angle) * basis2
            )
            circle_points.append(point)
        circle_points = np.array(circle_points)

        ins_x_lim.append([np.min(circle_points[:, 0]), np.max(circle_points[:, 0])])
        ins_y_lim.append([np.min(circle_points[:, 1]), np.max(circle_points[:, 1])])
        ins_z_lim.append([np.min(circle_points[:, 2]), np.max(circle_points[:, 2])])
        # Plot the circular projection
        ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='green' if i == 0 else 'red')

        # Plot the circular projection on the inset plot
        inset_ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='green' if i == 0 else 'red')

        # Plot the vector from the mixel
        if i == 0:
            ax.quiver(
                0,
                0,
                0,
                vector['cel_to_target_cartesian'][0],
                vector['cel_to_target_cartesian'][1],
                vector['cel_to_target_cartesian'][2],
                color='green',
                label=f"Start Time Local: {local_time_start}",
                arrow_length_ratio=0.1,
            )
            ax.quiver(
                0,
                0,
                0,
                observer_vec[0],
                observer_vec[1],
                observer_vec[2],
                color='black',
                label="Observer to Target",
                arrow_length_ratio=0.1,
            )
            ax.quiver(
                0,
                0,
                0,
                vector_dict['slope_1'][0],
                vector_dict['slope_1'][1],
                vector_dict['slope_1'][2],
                color='royalblue',
                label="Slope Option 1",
                arrow_length_ratio=0.1,
            )
            inset_ax.scatter(inter_1[0], inter_1[1], inter_1[2], c='royalblue', label="Intersection 1")
        else:
            ax.quiver(
                0,
                0,
                0,
                vector['cel_to_target_cartesian'][0],
                vector['cel_to_target_cartesian'][1],
                vector['cel_to_target_cartesian'][2],
                color='red',
                label=f"End Time Local: {local_time_end}",
                arrow_length_ratio=0.1,
            )
            ax.quiver(
                0,
                0,
                0,
                vector_dict['slope_2'][0],
                vector_dict['slope_2'][1],
                vector_dict['slope_2'][2],
                color='darkorange',
                label=f"Slope Option 2",
                arrow_length_ratio=0.1,
            )
            inset_ax.scatter(inter_2[0], inter_2[1], inter_2[2], c='darkorange', label="Intersection 2")

    # Set plot limits and labels
    ax.scatter([], [], [], c='royalblue', label="Intersection 1")
    ax.scatter([], [], [], c='darkorange', label="Intersection 2")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 0])
    ax.set_zlim([0, 1])
    ax.set_xlabel('X - East is Positive')
    ax.set_ylabel('Y - North is Positive')
    ax.set_zlabel('Z - Zenith is Positive')
    ax.set_title(f'Mixel {pixel} Bright to Dark on Unit Sphere with {celestial_object.capitalize()}')
    ax.set_aspect('equal')
    ax.legend(loc='lower center')

    # Set limits and labels for the inset plot
    inset_ax.set_xlim([np.min(ins_x_lim) - 0.01, np.max(ins_x_lim) + 0.01])
    inset_ax.set_ylim([np.min(ins_y_lim) - 0.01, np.max(ins_y_lim) + 0.01])
    inset_ax.set_zlim([np.min(ins_z_lim) - 0.01, np.max(ins_z_lim) + 0.01])
    # Reduce the number of tick marks on the inset plot
    inset_ax.set_xticks([np.round(np.min(ins_x_lim), 2), np.round(np.max(ins_x_lim), 2)])  # Fewer X-axis ticks
    inset_ax.set_yticks([np.round(np.min(ins_y_lim), 2), np.round(np.max(ins_y_lim), 2)])  # Fewer Y-axis ticks
    inset_ax.set_zticks([np.round(np.min(ins_z_lim), 2), np.round(np.max(ins_z_lim), 2)])  # Fewer Z-axis ticks

    inset_ax.set_title("Zoomed Projection")
    inset_ax.set_xlabel("X")
    inset_ax.set_ylabel("Y")
    inset_ax.set_zlabel("Z")
    # Save the plot to the output folder
    plot_file = os.path.join(output_folder, f"pixel_{pixel}_sky_plot_slope.png")

    if os.path.exists(plot_file):
        return f"File '{plot_file}' already exists. Skipping plot generation."
    else:
        plt.savefig(plot_file)
    plt.close()

    # plt.show()


def plotting_pixel_transition_vectors_decoupled(
    data_location, checkpoint_folder, checkpoint_data_file, checkpoint_plot_file, celestial_object, output_folder_img
):

    # Ensure the output folder exists
    os.makedirs(output_folder_img, exist_ok=True)

    checkpoint_data = load_checkpoint(checkpoint_folder, checkpoint_data_file)
    checkpoint_plots = load_checkpoint(checkpoint_folder, checkpoint_plot_file)
    if checkpoint_plots is None:
        checkpoint_plots = {"plotted_pixels": []}
    data = read_compressed_json(data_location)
    # Create a tqdm progress bar outside the loop
    progress_bar = tqdm(total=len(data), desc="Creating 3D Pixel Vector Plots")

    for pixel, _ in data.items():
        if pixel in checkpoint_data["pixels_without_data"]:
            logger.info("Skipping 3D plot for pixel %s without data.", pixel)
            # print(f"Skipping pixel {pixel} without data.")
            progress_bar.update(1)
            continue

        if isinstance(data[pixel], list):
            logger.info("Pixel %s incorrectly saved as having data", pixel)
            progress_bar.update(1)
            continue

        start_vector = data[pixel]['start_vector']
        end_vector = data[pixel]['end_vector']
        local_time_start = data[pixel]['start_time_Local']
        local_time_end = data[pixel]['end_time_Local']
        inter_1 = data[pixel]['intersection_1']
        inter_2 = data[pixel]['intersection_2']
        slope_1 = data[pixel]['slope_1']
        slope_2 = data[pixel]['slope_2']
        observer_vec = data[pixel]['observer_vector']
        # plotting vectors
        # Create the Positive Zenith side of the south half of a unit sphere for visualization of the horizon
        u = np.linspace(np.pi, 2 * np.pi, 100)
        v = np.linspace(0, np.pi / 2, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Initialize 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the unit sphere
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.15)  # Initialize 3D plot

        # Initialize inset 3D plot
        inset_ax = fig.add_axes([0.02, 0.05, 0.3, 0.3], projection='3d')  # Adjust position and size of inset plot

        # Plot the unit sphere on the inset plot
        inset_ax.plot_surface(x, y, z, color='lightblue', alpha=0.15)

        ins_x_lim = []
        ins_y_lim = []
        ins_z_lim = []
        # Plot the projections for each interpolated time
        for i, vector in enumerate([start_vector, end_vector]):

            # Convert angular size to radians and calculate angular radius
            angular_radius = vector['angular_size_radians'] / 2
            # Create a circle in 3D space for the projection
            num_points = 360  # Number of points to define the circle
            theta = np.linspace(0, 2 * np.pi, num_points)  # Angle around the circle

            # Find a basis for the plane perpendicular to the vector
            arbitrary_vector = (
                np.array([1, 0, 0])
                if not np.allclose(vector['cel_to_target_cartesian'], [1, 0, 0])
                else np.array([0, 1, 0])
            )
            basis1 = np.cross(vector['cel_to_target_cartesian'], arbitrary_vector)
            basis1 /= np.linalg.norm(basis1)  # Normalize

            basis2 = np.cross(vector['cel_to_target_cartesian'], basis1)
            basis2 /= np.linalg.norm(basis2)  # Normalize

            # Generate points on the circle
            circle_points = []
            for angle in theta:
                point = np.cos(angular_radius) * vector['cel_to_target_cartesian'] + np.sin(angular_radius) * (
                    np.cos(angle) * basis1 + np.sin(angle) * basis2
                )
                circle_points.append(point)
            circle_points = np.array(circle_points)

            ins_x_lim.append([np.min(circle_points[:, 0]), np.max(circle_points[:, 0])])
            ins_y_lim.append([np.min(circle_points[:, 1]), np.max(circle_points[:, 1])])
            ins_z_lim.append([np.min(circle_points[:, 2]), np.max(circle_points[:, 2])])
            # Plot the circular projection
            ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='green' if i == 0 else 'red')

            # Plot the circular projection on the inset plot
            inset_ax.plot(
                circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='green' if i == 0 else 'red'
            )

            # Plot the vector from the mixel
            if i == 0:
                ax.quiver(
                    0,
                    0,
                    0,
                    vector['cel_to_target_cartesian'][0],
                    vector['cel_to_target_cartesian'][1],
                    vector['cel_to_target_cartesian'][2],
                    color='green',
                    label=f"Start Time Local: {local_time_start}",
                    arrow_length_ratio=0.1,
                )
                ax.quiver(
                    0,
                    0,
                    0,
                    observer_vec[0],
                    observer_vec[1],
                    observer_vec[2],
                    color='black',
                    label="Observer to Target",
                    arrow_length_ratio=0.1,
                )
                ax.quiver(
                    0,
                    0,
                    0,
                    slope_1[0],
                    slope_1[1],
                    slope_1[2],
                    color='royalblue',
                    label="Slope Option 1",
                    arrow_length_ratio=0.1,
                )
                inset_ax.scatter(inter_1[0], inter_1[1], inter_1[2], c='royalblue', label="Intersection 1")
            else:
                ax.quiver(
                    0,
                    0,
                    0,
                    vector['cel_to_target_cartesian'][0],
                    vector['cel_to_target_cartesian'][1],
                    vector['cel_to_target_cartesian'][2],
                    color='red',
                    label=f"End Time Local: {local_time_end}",
                    arrow_length_ratio=0.1,
                )
                ax.quiver(
                    0,
                    0,
                    0,
                    slope_2[0],
                    slope_2[1],
                    slope_2[2],
                    color='darkorange',
                    label="Slope Option 2",
                    arrow_length_ratio=0.1,
                )
                inset_ax.scatter(inter_2[0], inter_2[1], inter_2[2], c='darkorange', label="Intersection 2")

        # Set plot limits and labels
        ax.scatter([], [], [], c='royalblue', label="Intersection 1")
        ax.scatter([], [], [], c='darkorange', label="Intersection 2")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 0])
        ax.set_zlim([0, 1])
        ax.set_xlabel('X - East is Positive')
        ax.set_ylabel('Y - North is Positive')
        ax.set_zlabel('Z - Zenith is Positive')
        ax.set_title(f'Mixel {pixel} Bright to Dark on Unit Sphere with {celestial_object.capitalize()}')
        ax.set_aspect('equal')
        ax.legend(loc='lower center')

        # Set limits and labels for the inset plot
        inset_ax.set_xlim([np.min(ins_x_lim) - 0.01, np.max(ins_x_lim) + 0.01])
        inset_ax.set_ylim([np.min(ins_y_lim) - 0.01, np.max(ins_y_lim) + 0.01])
        inset_ax.set_zlim([np.min(ins_z_lim) - 0.01, np.max(ins_z_lim) + 0.01])
        # Reduce the number of tick marks on the inset plot
        inset_ax.set_xticks([np.round(np.min(ins_x_lim), 2), np.round(np.max(ins_x_lim), 2)])  # Fewer X-axis ticks
        inset_ax.set_yticks([np.round(np.min(ins_y_lim), 2), np.round(np.max(ins_y_lim), 2)])  # Fewer Y-axis ticks
        inset_ax.set_zticks([np.round(np.min(ins_z_lim), 2), np.round(np.max(ins_z_lim), 2)])  # Fewer Z-axis ticks

        inset_ax.set_title("Zoomed Projection")
        inset_ax.set_xlabel("X")
        inset_ax.set_ylabel("Y")
        inset_ax.set_zlabel("Z")
        # Save the plot to the output folder
        plot_file = os.path.normpath(os.path.join(output_folder_img, f"pixel_{pixel}_sky_plot_slope.png"))
        plt.savefig(plot_file)
        plt.close()
        checkpoint_plots["plotted_pixels"].append(pixel)
        save_checkpoint(checkpoint_folder, checkpoint_plot_file, checkpoint_plots)
        progress_bar.update(1)
    progress_bar.close()


def trilaterate(positions: np.ndarray, radii: np.ndarray, raise_on_no_solution: bool = False) -> np.ndarray:
    """
    Trilateration algorithm to find the intersection points of three spheres in 3D space.

    :param positions: A 3x3 array where each row represents the (x, y, z) coordinates of a sphere's center.
    :param radii: A 3-element array representing the radius of each sphere.
    :param raise_on_no_solution: If True, raises an error when no exact solution exists. If False, returns a close-enough solution.
    :return: A 2x3 array of intersection points (two solutions), or a 1x3 array if there is only one solution or no exact solution.

    Notes:
    - This implementation is based on the mathematical derivation of trilateration.
    - Modified from https://stackoverflow.com/a/18654302/313768
    """
    # Extract radii and positions
    radius1, radius2, radius3 = radii
    center1, center2, center3 = positions

    # Step 1: Compute inter-point vectors
    vector21 = center2 - center1  # Vector from center1 to center2
    vector31 = center3 - center1  # Vector from center1 to center3
    distance12 = np.linalg.norm(vector21)  # Distance between center1 and center2
    distance13 = np.linalg.norm(vector31)  # Distance between center1 and center3
    distance23 = np.linalg.norm(center3 - center2)  # Distance between center2 and center3

    # Step 2: Check for degenerate cases (e.g., overlapping spheres)
    if np.allclose(center1, center2) and np.isclose(radius1, radius2):
        raise ValueError("Two spheres have the same center and radius, resulting in an ambiguous or infinite solution.")
    if np.allclose(center1, center3) and np.isclose(radius1, radius3):
        raise ValueError("Two spheres have the same center and radius, resulting in an ambiguous or infinite solution.")
    if np.allclose(center2, center3) and np.isclose(radius2, radius3):
        raise ValueError("Two spheres have the same center and radius, resulting in an ambiguous or infinite solution.")

    # Check for non-overlapping spheres
    if distance12 > radius1 + radius2 or distance12 < abs(radius1 - radius2):
        raise ValueError("Spheres 1 and 2 do not overlap, no solution exists.")
    if distance13 > radius1 + radius3 or distance13 < abs(radius1 - radius3):
        raise ValueError("Spheres 1 and 3 do not overlap, no solution exists.")
    if distance23 > radius2 + radius3 or distance23 < abs(radius2 - radius3):
        raise ValueError("Spheres 2 and 3 do not overlap, no solution exists.")

    # Step 3: Compute basis vectors for the coordinate system
    unit_vector_u = vector21 / distance12  # Unit vector along vector21
    projection_i = unit_vector_u.dot(vector31)  # Projection of vector31 onto unit_vector_u
    vector_v = vector31 - unit_vector_u * projection_i  # Orthogonal component of vector31
    vector_v /= np.linalg.norm(vector_v)  # Normalize vector_v
    projection_j = vector_v.dot(vector31)  # Projection of vector31 onto vector_v
    unit_vector_w = np.cross(unit_vector_u, vector_v)  # Unit vector orthogonal to both unit_vector_u and vector_v

    # Step 4: Solve for the x and y coordinates in the projected space
    x = 0.5 / distance12 * (radius1**2 - radius2**2 + distance12**2)
    y = 0.5 / projection_j * (radius1**2 - radius3**2 - 2 * projection_i * x + projection_i**2 + projection_j**2)
    radicand = radius1**2 - x**2 - y**2  # Compute the radicand for the z-coordinate

    # Step 5: Handle cases where the radicand is negative (no exact solution)
    if radicand < 0:
        if raise_on_no_solution:
            raise ValueError(f"Negative radicand {radicand}: no exact solutions exist.")
        return (center1 + unit_vector_u * x + vector_v * y)[np.newaxis, :]  # Return a close-enough solution

    # Step 6: Compute the z-coordinate and intersection points
    z = np.sqrt(radicand)  # Compute the z-coordinate
    offset_z = unit_vector_w * z  # Offset in the z-direction
    solution_a = center1 + unit_vector_u * x + vector_v * y + offset_z  # First intersection point
    solution_b = center1 + unit_vector_u * x + vector_v * y - offset_z  # Second intersection point

    # Step 7: Return the solutions
    return np.stack((solution_a, solution_b))  # Return both intersection points as a 2x3 array


def lat_long_to_decimal(input):
    decimal = input[0] + input[1] / 60 + input[2] / 3600
    return decimal


def maximum_frame_range(frame_range):
    differences = []
    for i, _ in enumerate(frame_range):
        if i == len(frame_range) - 1:
            continue
        else:
            if frame_range[i][0] != frame_range[i + 1][0]:
                differences.append(
                    (frame_range[i], frame_range[i + 1], np.abs(frame_range[i + 1][1] - frame_range[i][1]))
                )
    max_difference = max(differences, key=lambda item: item[2])
    frame_range_return = [max_difference[0], max_difference[1]]
    return frame_range_return, max_difference[2]


def frame_number_from_img_name(image_name_str):
    # returns the integer number of a frame given the following format
    # "DSC_2832-09170.png" where DSC_2832 is the video source and "09170"
    # is the frame number
    _, tail = os.path.split(image_name_str)
    _, frame = re.findall(r'\d+', tail)
    return int(frame)


def define_observation_time(video_time, time_offset):
    video_time = video_time + time_offset
    observation_time = video_time.astimezone(timezone.utc)
    # Define observation time
    ts = skf.load.timescale()
    if isinstance(observation_time, skf.Time):
        pass
    elif isinstance(observation_time, datetime):
        # Extract components from the datetime object
        observation_time = ts.utc(
            observation_time.year,
            observation_time.month,
            observation_time.day,
            observation_time.hour,
            observation_time.minute,
            observation_time.second,
        )
    else:
        # If observation_time is already a tuple, unpack it directly
        observation_time = ts.utc(*observation_time)
    return observation_time


# Example usage
if __name__ == "__main__":
    # File Locations
    json_data_location = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/7_pixel_timing_interrogation/time_history_transition_parallel_facet.json.gz"
    output_folder_img = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/8_pixel_vector_information/pixel_vector_plots"
    output_folder_vector = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/8_pixel_vector_information/parallel"
    output_vector_data_name = "pixel_vector_information_wslope.json.gz"
    video_file_path = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/DSC_0025.MOV"  # Video Path Used to Generate Frames
    checkpoint_dir = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/0_checkpoints"
    checkpoint_file_data = "pixel_vector_checkpoint_info_parallel.json"
    checkpoint_file_3d_plots = "pixel_vector_3D plotting_checkpoint.json"

    celestial_object = "sun"
    metadata = extract_video_metadata_exiftool(video_file_path)

    # Define observer location Example ≈NSTTF Tower 260 Level Balcony West Side
    observer_lat = (34, 57, 44.56)  # (degree, minute, second) negative degree for south
    observer_long = (-106, 30, 34.90)  # (degree, minute, second) negative degree for west
    observer_elevation = 1755.648 + 1.2192
    observer_location = (lat_long_to_decimal(observer_lat), lat_long_to_decimal(observer_long), observer_elevation)

    # Define target location Example ≈Sun Data Marker 2 In front of 5E8
    target_lat = (34, 57, 45.92)  # (degree, minute, second) negative degree for south
    target_long = (-106, 30, 31.88)  # (degree, minute, second) negative degree for west
    target_elevation = 1706.88
    target_location = (
        lat_long_to_decimal(target_lat),
        lat_long_to_decimal(target_long),
        target_elevation,
    )  # Example target location (latitude, longitude, elevation in meters)

    '''
    # Define observer location Example ≈NSTTF Tower 260 Level Balcony East Side
    observer_lat = (34, 57, 44.56)  # (degree, minute, second) negative degree for south
    observer_long = (-106, 30, 34.65)  # (degree, minute, second) negative degree for west
    observer_elevation = 1755.648 + 1.2192
    observer_location = (lat_long_to_decimal(observer_lat), lat_long_to_decimal(observer_long), observer_elevation)

    # Define target location Example ≈NSTTF Heliostat 5E9
    target_lat = (34, 57, 46.08)  # (degree, minute, second) negative degree for south
    target_long = (-106, 30, 31.43)  # (degree, minute, second) negative degree for west
    target_elevation = 1706.88
    target_location = (
        lat_long_to_decimal(target_lat),
        lat_long_to_decimal(target_long),
        target_elevation,
    )  # Example target location (latitude, longitude, elevation in meters)


    extract_pixel_timing_and_celestial_vectors(
        celestial_object_name=celestial_object,
        target_location=target_location,
        observer_location=observer_location,
        camera_time_shift=camera_time_shift,
        data_location=json_data_location,
        video_metadata=metadata,
        output_folder_vector=output_folder_vector,
        output_json_name=output_vector_data_name,
        checkpoint_folder=checkpoint_dir,
        checkpoint_file=checkpoint_file_data,
    )
    '''

    camera_time_shift = timedelta(hours=5, minutes=50, seconds=0)

    extract_pixel_timing_and_celestial_vectors_parallel(
        celestial_object_name=celestial_object,
        target_location=target_location,
        observer_location=observer_location,
        camera_time_shift=camera_time_shift,
        data_location=json_data_location,
        video_metadata=metadata,
        output_folder_vector=output_folder_vector,
        output_json_name=output_vector_data_name,
        checkpoint_folder=checkpoint_dir,
        checkpoint_file=checkpoint_file_data,
        batch_size=5000,
    )

    plotting_pixel_transition_vectors_decoupled(
        data_location=os.path.join(output_folder_vector, output_vector_data_name),
        checkpoint_folder=checkpoint_dir,
        checkpoint_data_file=checkpoint_file_data,
        checkpoint_plot_file=checkpoint_file_3d_plots,
        celestial_object=celestial_object,
        output_folder_img=output_folder_img,
    )
