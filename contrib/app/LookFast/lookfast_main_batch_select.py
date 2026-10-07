import os
import tkinter as tk
from tkinter import filedialog
from logging import DEBUG, ERROR

import opencsp.common.lib.tool.file_tools as ft

# import opencsp.app.lookback.lookback_tools as lbt
import contrib.app.LookFast.lookback_tools as lbt

from contrib.app.LookFast.lookfast_main_VCLI_argparse import wrapped_launch

# Specify the folder where the log file should be saved
logger = lbt.logging_setup(
    log_folder=ft.join(os.getcwd(), "error_logs"),
    log_file_name="error_log_lookfast_main_batch_select.txt",
    log_type=DEBUG,
)


def select_files(window_title="Select Files", file_types=None):
    """
    Opens a file dialog to select multiple files, with customizable window title and file types.

    Parameters:
        window_title (str): The title of the file dialog window (default is "Select Files").
        file_types (str): A string of file types in the format "*.ext1 *.ext2", e.g., "*.jpg *.png".
                         If None, defaults to allowing all files.

    Returns:
        list: A list of normalized file paths, or an empty list [] if no files were selected.
    """
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Parse file types into the required format for filedialog
    if file_types:
        # Split into a list, e.g., ['*.jpg', '*.png']
        extensions = file_types.lower().split()
        # Tkinter requires a tuple of tuples: (("Description", "*.ext1 *.ext2"), ...)
        # We join them back with spaces inside the tuple for the filter to work correctly
        file_types_list = [("Custom File Types", " ".join(extensions)), ("All Files", "*.*")]
    else:
        file_types_list = [("All Files", "*.*")]

    # Open the file dialog (notice the plural 'askopenfilenames')
    file_paths = filedialog.askopenfilenames(title=window_title, filetypes=file_types_list)

    if file_paths:
        logger.info("Files Selected: %s", list(file_paths))
        # Normalize all paths in the tuple and return them as a list
        result = [os.path.normpath(path) for path in file_paths]
    else:
        logger.info("Files Selected: None")
        result = []

    root.destroy()
    return result


if __name__ == "__main__":
    settings_files = select_files(
        window_title="Select Multiple '.ini' files for LOOKFAST Processing", file_types="*.ini"
    )

    for file in settings_files:
        logger.info("Processing File: %s", file)
        wrapped_launch(file)
        logger.info("Processing Complete for File: %s", file)
