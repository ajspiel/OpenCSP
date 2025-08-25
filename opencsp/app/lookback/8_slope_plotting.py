import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import gzip
from zoneinfo import ZoneInfo
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import subprocess


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


# Extract slope data and pixel coordinates
def extract_data(data):
    slope_1 = []
    slope_2 = []
    pixel_coords = []
    for pixel, details in data.items():
        for entry in details:
            if "slope_1" in entry and "slope_2" in entry:
                slope_1.append(entry["slope_1"])
                slope_2.append(entry["slope_2"])
                pixel_coords.append(eval(pixel))  # Convert string "(x, y)" to tuple
    return np.array(pixel_coords), np.array(slope_1), np.array(slope_2)


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


# Main execution
if __name__ == "__main__":
    # File paths
    json_filepath = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/8_pixel_vector_information/pixel_vector_information_wslope.json"
    image_filepath = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/4_coverage_map/traditional_compiled_binary_map_50.png"  # Replace with your source image path

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
