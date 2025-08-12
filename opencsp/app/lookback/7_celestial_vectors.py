import re
import os
import json
import gzip
from zoneinfo import ZoneInfo
from datetime import datetime, timezone, timedelta
import skyfield.api as skf
import numpy as np
from tqdm import tqdm
import subprocess
import matplotlib.pyplot as plt


def save_checkpoint(checkpoint_folder, checkpoint_file_name, checkpoint_data):
    """
    Saves the checkpoint data to a JSON file.

    Parameters:
        checkpoint_file_name (str): Path to the checkpoint file.
        checkpoint_data (dict): Dictionary containing checkpoint information.

    Returns:
        None
    """
    with open(os.path.join(checkpoint_folder, checkpoint_file_name), 'w') as f:
        json.dump(checkpoint_data, f, indent=4)
    print(f"Checkpoint saved to {os.path.join(checkpoint_folder, checkpoint_file_name)}")


def load_checkpoint(checkpoint_folder, checkpoint_file_name):
    """
    Loads the checkpoint data from a JSON file.

    Parameters:
        checkpoint_file_name (str): Path to the checkpoint file.

    Returns:
        dict: Dictionary containing checkpoint information.
    """
    if os.path.exists(os.path.join(checkpoint_folder, checkpoint_file_name)):
        with open(os.path.join(checkpoint_folder, checkpoint_file_name), 'r') as f:
            checkpoint_data = json.load(f)
        print(f"Checkpoint loaded from {os.path.join(checkpoint_folder, checkpoint_file_name)}")
        return checkpoint_data
    else:
        return None


def custom_serializer(obj):
    """
    Custom serializer to handle numpy arrays and datetime objects.
    Converts unsupported types into JSON-compatible formats.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy array to list
    elif isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to ISO 8601 string
    elif isinstance(obj, set):
        return list(obj)  # Convert set to list
    raise TypeError(f"Type {type(obj)} not serializable")


def custom_deserializer(obj):
    """
    Custom deserializer to handle numpy arrays and datetime objects.
    Converts JSON-compatible formats back into their original types.
    """
    for key, value in obj.items():
        if isinstance(value, list) and all(isinstance(i, (int, float)) for i in value):
            try:
                obj[key] = np.array(value)  # Convert list back to numpy array
            except ValueError:
                pass  # If conversion fails, leave as list
        elif isinstance(value, str):
            try:
                obj[key] = datetime.fromisoformat(value)  # Convert ISO 8601 string back to datetime
            except ValueError:
                pass  # If conversion fails, leave as string
    return obj


def read_json(file_path):
    """
    Reads a regular JSON file and returns the data.

    Parameters:
        file_path (str): Path to the .json file.

    Returns:
        dict or list: Data from the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f, object_hook=custom_deserializer)
        return data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
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
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, default=custom_serializer)
        print(f"Data written to {file_path}")
    except Exception as e:
        print(f"Error writing file {file_path}: {e}")


def read_compressed_json(file_path):
    """
    Reads a compressed JSON file (.json.gz) and returns the data.

    Parameters:
        file_path (str): Path to the .json.gz file.

    Returns:
        list: List of dictionaries containing the data from the file.
    """
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f, object_hook=custom_deserializer)
        return data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def write_compressed_json(data, file_path):
    """
    Writes data to a compressed JSON file (.json.gz).

    Parameters:
        data (dict or list): Data to write to the file.
        file_path (str): Path to the .json.gz file.

    Returns:
        None
    """
    try:
        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=4, default=custom_serializer)
        print(f"Data written to {file_path}")
    except Exception as e:
        print(f"Error writing file {file_path}: {e}")


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
    except Exception as e:
        print(f"Error extracting metadata with exiftool: {e}")
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
    observer_to_target_bary = observer_position.at(observation_time)
    observer_to_target = observer_to_target_bary.observe(target_position_vec)
    observer_to_target_apparent = observer_to_target.apparent()

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
        observer_to_target_apparent.frame_xyz(target).km[1],
        observer_to_target_apparent.frame_xyz(target).km[0],
        observer_to_target_apparent.frame_xyz(target).km[2],
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
        "observer_to_target": NSTTF_coord_otta / np.linalg.norm(NSTTF_coord_otta),
        "angular_size_radians": angular_size_radians,
    }


def extract_pixel_timing_and_celestial_vectors(
    celestial_object_name,
    target_location,
    observer_location,
    camera_time_shift,
    data_location,
    video_metadata,
    output_folder_img,
    output_folder_vector,
    output_json_name,
):
    video_start_time = datetime.strptime(video_metadata['creation_date'], "%Y:%m:%d %H:%M:%S")
    abq_tz = ZoneInfo("America/Denver")
    video_start_time = video_start_time.replace(tzinfo=abq_tz)
    # Read in pixel timing information
    data = read_compressed_json(data_location)

    for pixel, transitions in data.items():
        frame_range = []
        for transition in transitions:
            if transition["transition"] == "bright":
                frame_range.append(tuple((1, frame_number_from_img_name(transition['to_frame']))))
            elif transition["transition"] == "dark":
                frame_range.append(tuple((0, frame_number_from_img_name(transition['to_frame']))))

        if len(frame_range) != 2:
            raise ValueError(f"More than one transition cycle detected on {pixel}")
        else:
            frame_diff = frame_range[1][1] - frame_range[0][1]
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
        intersection_1, intersection_2 = trilaterate(points, radii)

        # TODO Add observer location and intersection vector addition to estimate slopes.
        # TODO Plot Slopes and observer location as vectors for sanity checking

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
            }
        )

    for pixel, dicts in data.items():
        plot_num = 0
        if plot_num == 0:
            plotting_pixel_transition_vectors(pixel, dicts[2], celestial_object_name, output_folder=output_folder_img)
        else:
            pass
        plot_num += 1

    # write_compressed_json(data=data, file_path=os.path.join(output_folder_vector, output_json_name))
    # write_json(data=data, file_path=os.path.join(output_folder_vector, output_json_name[:-3]))


def plotting_pixel_transition_vectors(pixel, vector_dict, celestial_object, output_folder):
    start_vector = vector_dict['start_vector']
    end_vector = vector_dict['end_vector']
    local_time_start = vector_dict['start_time_Local']
    local_time_end = vector_dict['end_time_Local']
    inter_1 = vector_dict['intersection_1']
    inter_2 = vector_dict['intersection_2']
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
    plot_file = os.path.join(output_folder, f"pixel_{pixel}_sky_plot.png")
    if os.path.exists(plot_file):
        return f"File '{plot_file}' already exists. Skipping plot generation."
    else:
        plt.savefig(plot_file)
    plt.close()
    # plt.show()


def trilaterate(p: np.ndarray, norms: np.ndarray, fatal_radical: bool = False) -> np.array:  # 3*3  # *3  # 2*3
    """
    https://en.wikipedia.org/wiki/True-range_multilateration#Three_Cartesian_dimensions,_three_measured_slant_ranges

    :param p 3*3 array where the first axis is the observation and the second axis is xyz
    :param norms 3-array, the distance from each observation to an unknown centroid
    :param fatal_radical True to throw if no solution; false to assume a close-enough solution
    :return a 2x3 array of centroid solutions, or 1x3 if there is either one solution, or
            fatal_radical is false and there are no exact solutions

    Modified from https://stackoverflow.com/a/18654302/313768
    """
    r1, r2, r3 = norms
    p1, p2, p3 = p

    # Inter-point vectors
    p21 = p2 - p1
    p31 = p3 - p1
    d = np.linalg.norm(p21)

    # Basis vectors
    u = p21 / d
    i = u.dot(p31)
    v = p31 - u * i
    v /= np.linalg.norm(v)
    j = v.dot(p31)
    w = np.cross(u, v)

    # Solution dimensions, still in projected space
    x = 0.5 / d * (r1 * r1 - r2 * r2 + d * d)
    y = 0.5 / j * (r1 * r1 - r3 * r3 - 2 * i * x + i * i + j * j)
    radicand = r1 * r1 - x * x - y * y

    # Solution vectors in original space
    sxy = p1 + u * x + v * y  # First two dimensions, missing 'z'

    if radicand < 0:
        if fatal_radical:
            raise ValueError(f'Negative radicand {radicand}: no exact solutions')
    if radicand <= 0:
        return sxy[np.newaxis, :]

    z = np.sqrt(radicand)
    sz = w * z  # 'z', prior to sign change
    sa = sxy + sz  # First solution
    sb = sxy - sz  # Second solution
    return np.stack((sa, sb))  # Both solutions: 2*3


def lat_long_to_decimal(input):
    decimal = input[0] + input[1] / 60 + input[2] / 3600
    return decimal


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
    # "//snl//Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/7_pixel_timing_interrogation/pixel_timing_plots/pixel_(100, 1100)_timing_plot_data.json.gz"
    json_data_location = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/7_pixel_timing_interrogation/time_history_transition_parallel_facet.json.gz"
    output_folder_img = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/7_pixel_timing_interrogation/pixel_timing_plots"
    output_folder_vector = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/8_pixel_vector_information"
    output_vector_data_name = "pixel_vector_information.json.gz"
    video_file_path = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/DSC_2832.MOV"  # Video Path Used to Generate Frames
    metadata = extract_video_metadata_exiftool(video_file_path)

    # Define observer location Example ≈NSTTF Tower 260 Level Balcony
    observer_lat = (34, 57, 44.56)  # (degree, minute, second) negative degree for south or west
    observer_long = (-106, 30, 34.70)  # (degree, minute, second) negative degree for south or west
    observer_elevation = 1755.648
    observer_location = (lat_long_to_decimal(observer_lat), lat_long_to_decimal(observer_long), observer_elevation)

    # Define target location Example ≈NSTTF Heliostat 5E9
    target_lat = (34, 57, 46.08)  # (degree, minute, second) negative degree for south or west
    target_long = (-106, 30, 31.43)  # (degree, minute, second) negative degree for south or west
    target_elevation = 1706.88
    target_location = (
        lat_long_to_decimal(target_lat),
        lat_long_to_decimal(target_long),
        target_elevation,
    )  # Example target location (latitude, longitude, elevation in meters)

    camera_time_shift = timedelta(hours=0, minutes=0, seconds=0)

    extract_pixel_timing_and_celestial_vectors(
        "moon",
        target_location=target_location,
        observer_location=observer_location,
        camera_time_shift=camera_time_shift,
        data_location=json_data_location,
        video_metadata=metadata,
        output_folder_img=output_folder_img,
        output_folder_vector=output_folder_vector,
        output_json_name=output_vector_data_name,
    )
