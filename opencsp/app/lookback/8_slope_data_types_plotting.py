import json
import time
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import gzip
from zoneinfo import ZoneInfo
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import subprocess

# Specify the folder where the log file should be saved
log_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/0_checkpoints/error_logs"  # Replace with your desired folder path
log_file = os.path.join(log_folder, "error_log_slope_plotting.txt")

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


# Extract slope data and pixel coordinates
def extract_data(data):
    slope_1 = []
    slope_2 = []
    pixel_coords = []
    pixel_no_slope = []
    pixel_no_overlap = []
    slope_no_overlap = []
    for pixel, details in data.items():
        if isinstance(details, dict):
            if "slope_1" in details or "slope_2" in details:
                if details["slope_1"].size < 3 or details["slope_2"].size < 3:
                    pixel_no_slope.append(eval(pixel))
                    continue
                elif np.array_equal(details["slope_1"], details["slope_2"]):
                    pixel_no_overlap.append(eval(pixel))
                    slope_no_overlap.append(details["slope_1"])
                else:
                    slope_1.append(details["slope_1"])
                    slope_2.append(details["slope_2"])
                    pixel_coords.append(eval(pixel))  # Convert string "(x, y)" to tuple
            else:
                continue
    return (
        np.array(pixel_coords),
        np.array(pixel_no_slope),
        np.array(pixel_no_overlap),
        np.array(slope_1),
        np.array(slope_2),
        np.array(slope_no_overlap),
    )


# Compute slope magnitudes
def compute_magnitude(slope_data):
    return np.linalg.norm(slope_data, axis=1)


# Generate contour plots
def plot_contours(pixel_coords, values, title, cmap="viridis"):
    x = pixel_coords[:, 0]
    y = pixel_coords[:, 1]
    z = values

    # Create grid for contour plot
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = plt.tricontourf(x, y, z, levels=100, cmap=cmap).get_array()

    plt.figure(figsize=(8, 6))
    plt.tricontourf(x, y, z, levels=100, cmap=cmap)
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    # plt.show()


# Overlay slope magnitude on source image
def overlay_on_image(pixel_coords, values, image_path, cmap="viridis"):
    # Load the source image
    img = plt.imread(image_path)

    # Plot the image
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap="gray")
    plt.scatter(pixel_coords[:, 1], pixel_coords[:, 0], c=values, cmap=cmap, s=10)
    plt.colorbar(label="Slope Magnitude")
    plt.title("Slope Magnitude Overlay")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    # plt.show()


def disp_pixels_with_data(data_path, chkpt_data_path, image_path):
    data = read_compressed_json(data_path)
    chkpt_data = read_json(chkpt_data_path)
    # chkpt_plot_data = read_json(chkpt_plot_path)

    pixels, pixel_no_slope, pixel_no_overlap, _, _, _ = extract_data(data)
    all_pixels = []
    for pix in chkpt_data["processed_pixels"]:
        x, y = eval(pix)
        all_pixels.append([x, y])

    all_pixels = np.array(all_pixels)
    # Load the source image
    img = plt.imread(image_path)

    # Plot the image
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap="gray")
    plt.scatter(all_pixels[:, 1], all_pixels[:, 0], c="r", s=1, marker="s", alpha=0.1, label="All Processed Pixels")
    '''
    plt.scatter(pixels[:, 1], pixels[:, 0], c="b", s=1, marker="s", alpha=0.5, label="Pixels with Calculated Slopes")
    plt.scatter(
        pixel_no_slope[:, 1], pixel_no_slope[:, 0], c="m", s=1, marker="s", alpha=0.5, label="Pixels without Slopes"
    )
    plt.scatter(
        pixel_no_overlap[:, 1],
        pixel_no_overlap[:, 0],
        c="g",
        s=1,
        marker="s",
        alpha=0.5,
        label="Pixels without Overlap",
    )
    '''

    plt.title("Data Coverage Map from Full Sampeled Set")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.legend()
    plt.show()
    print("this line")


# Main execution
if __name__ == "__main__":
    # File paths
    compressed_json_filepath = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/8_pixel_vector_information/debug/pixel_vector_information_wslope_debug.json.gz"
    image_filepath = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/4_coverage_map/traditional_compiled_binary_map_50.png"  # Replace with your source image path

    checkpoint_dir = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/0_checkpoints"
    checkpoint_file_data = "pixel_vector_checkpoint_info_parallel_debug.json"
    # checkpoint_file_3d_plots = "pixel_vector_3D plotting_checkpoint.json"

    disp_pixels_with_data(
        data_path=compressed_json_filepath,
        chkpt_data_path=os.path.join(checkpoint_dir, checkpoint_file_data),
        image_path=image_filepath,
    )

    '''
    # Load and process data
    data = read_json(json_filepath)
    pixel_coords, slope_1, slope_2 = extract_data(data)
    slope_magnitude = compute_magnitude(slope_1)

    # Contour plots
    plot_contours(pixel_coords, slope_magnitude, "Surface Normal Magnitude Global")
    plot_contours(pixel_coords, slope_1[:, 0], "Surface Normal X Component")
    plot_contours(pixel_coords, slope_1[:, 1], "Surface Normal Y Component")
    plot_contours(pixel_coords, slope_1[:, 2], "Surface Normal Z Component")

    # Overlay slope magnitude on source image
    overlay_on_image(pixel_coords, slope_magnitude, image_filepath)
    plt.show()
    print("done")
    # TODO I need to think of a way to visualize the slope map generated,
    # because all of the slopes are normalized...
    '''
