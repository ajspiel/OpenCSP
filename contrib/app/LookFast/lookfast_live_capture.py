#!/usr/bin/env python3
"""
lookfast_live_capture.py

A flexible computer vision script for simultaneous multi-camera capture, display, and processing.
Features:
- Capture from multiple cameras simultaneously
- Display scaled-down video streams on screen
- Separate GUI control window for each camera's settings
- Multiple modes: Live display, Live + GPU processing, and Recording
- Configurable output locations for each camera

Author: OpenCSP
Date: 2026
"""

import cv2
import numpy as np
import threading
import queue
import time
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, List, Tuple, Optional, Any
import argparse
import json
import logging
from datetime import datetime
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CameraSettings:
    """Class to manage settings for an individual camera."""
    
    def __init__(self, camera_id: int = 0, name: str = "Camera"):
        self.camera_id = camera_id
        self.name = name
        self.width = 1920
        self.height = 1080
        self.fps = 30
        self.brightness = 50
        self.contrast = 50
        self.saturation = 50
        self.hue = 50
        self.gain = 50
        self.exposure = 50
        self.focus = 50
        self.enable_processing = False
        self.enable_recording = False
        self.output_directory = "/tmp"
        self.display_scale = 0.5
        self.show_controls = True
        self.processing_algorithm = "edge_detection"
        self.recording_format = "mp4"
        self.max_frames = 1000
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            'camera_id': self.camera_id,
            'name': self.name,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'saturation': self.saturation,
            'hue': self.hue,
            'gain': self.gain,
            'exposure': self.exposure,
            'focus': self.focus,
            'enable_processing': self.enable_processing,
            'enable_recording': self.enable_recording,
            'output_directory': self.output_directory,
            'display_scale': self.display_scale,
            'show_controls': self.show_controls,
            'processing_algorithm': self.processing_algorithm,
            'recording_format': self.recording_format,
            'max_frames': self.max_frames
        }
    
    def from_dict(self, settings_dict: Dict[str, Any]):
        """Load settings from dictionary."""
        for key, value in settings_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class CameraCapture:
    """Class to handle video capture from a single camera."""
    
    def __init__(self, camera_id: int, settings: CameraSettings):
        self.camera_id = camera_id
        self.settings = settings
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.is_running = False
        self.capture_thread = None
        self.processing_thread = None
        self.recording_thread = None
        self.writer = None
        self.frame_count = 0
        self.start_time = None
        self.last_frame_time = None
        
    def initialize(self) -> bool:
        """Initialize the camera capture."""
        try:
            # Try to open the camera
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.settings.fps)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.settings.brightness / 100.0)
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.settings.contrast / 100.0)
            self.cap.set(cv2.CAP_PROP_SATURATION, self.settings.saturation / 100.0)
            self.cap.set(cv2.CAP_PROP_HUE, self.settings.hue / 100.0)
            self.cap.set(cv2.CAP_PROP_GAIN, self.settings.gain / 100.0)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.settings.exposure / 100.0)
            self.cap.set(cv2.CAP_PROP_FOCUS, self.settings.focus / 100.0)
            
            # Get actual settings
            self.settings.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.settings.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.settings.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera {self.camera_id} initialized: {self.settings.width}x{self.settings.height} @ {self.settings.fps:.1f}fps")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera {self.camera_id}: {str(e)}")
            return False
    
    def start(self):
        """Start the camera capture thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        if self.settings.enable_processing:
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
        
        if self.settings.enable_recording:
            self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
            self.recording_thread.start()
    
    def stop(self):
        """Stop the camera capture."""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        if self.writer:
            self.writer.release()
        
        logger.info(f"Camera {self.camera_id} stopped")
    
    def _capture_loop(self):
        """Main capture loop."""
        self.start_time = time.time()
        self.frame_count = 0
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning(f"Camera {self.camera_id}: Failed to capture frame")
                    time.sleep(0.01)
                    continue
                
                # Put frame in queue
                if not self.frame_queue.full():
                    self.frame_queue.put((frame, time.time()))
                
                self.frame_count += 1
                self.last_frame_time = time.time()
                
                # Calculate actual FPS
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
                    logger.debug(f"Camera {self.camera_id}: {actual_fps:.1f} fps")
                
            except Exception as e:
                logger.error(f"Camera {self.camera_id} capture error: {str(e)}")
                time.sleep(0.1)
    
    def _processing_loop(self):
        """Processing loop for GPU/CPU processing."""
        while self.is_running and self.settings.enable_processing:
            try:
                if not self.frame_queue.empty():
                    frame, timestamp = self.frame_queue.get(timeout=0.1)
                    
                    # Apply processing based on selected algorithm
                    processed_frame = self._apply_processing(frame)
                    
                    # Put processed frame back in queue
                    if not self.frame_queue.full():
                        self.frame_queue.put((processed_frame, timestamp))
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Camera {self.camera_id} processing error: {str(e)}")
                time.sleep(0.1)
    
    def _apply_processing(self, frame: np.ndarray) -> np.ndarray:
        """Apply selected processing algorithm."""
        if self.settings.processing_algorithm == "edge_detection":
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 100, 200)
            # Convert back to 3-channel for display
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif self.settings.processing_algorithm == "blur":
            return cv2.GaussianBlur(frame, (15, 15), 0)
        elif self.settings.processing_algorithm == "grayscale":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            return frame
    
    def _recording_loop(self):
        """Recording loop to save frames to disk."""
        output_file = self._get_output_filename()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.settings.output_directory, exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if self.settings.recording_format == "mp4" else cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(
            output_file, 
            fourcc, 
            self.settings.fps, 
            (self.settings.width, self.settings.height)
        )
        
        if not self.writer.isOpened():
            logger.error(f"Camera {self.camera_id}: Failed to open video writer for {output_file}")
            return
        
        logger.info(f"Camera {self.camera_id}: Recording to {output_file}")
        
        recorded_frames = 0
        while self.is_running and self.settings.enable_recording and recorded_frames < self.settings.max_frames:
            try:
                if not self.frame_queue.empty():
                    frame, timestamp = self.frame_queue.get(timeout=0.1)
                    self.writer.write(frame)
                    recorded_frames += 1
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Camera {self.camera_id} recording error: {str(e)}")
                time.sleep(0.1)
        
        logger.info(f"Camera {self.camera_id}: Recording stopped. Saved {recorded_frames} frames.")
    
    def _get_output_filename(self) -> str:
        """Generate output filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(
            self.settings.output_directory,
            f"camera_{self.camera_id}_{timestamp}.{self.settings.recording_format}"
        )
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Get the latest frame from the queue."""
        try:
            if not self.frame_queue.empty():
                return self.frame_queue.get(timeout=0.1)
            return None
        except queue.Empty:
            return None
    
    def update_settings(self, new_settings: CameraSettings):
        """Update camera settings."""
        # Update settings that can be changed dynamically
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, new_settings.brightness / 100.0)
        self.cap.set(cv2.CAP_PROP_CONTRAST, new_settings.contrast / 100.0)
        self.cap.set(cv2.CAP_PROP_SATURATION, new_settings.saturation / 100.0)
        self.cap.set(cv2.CAP_PROP_HUE, new_settings.hue / 100.0)
        self.cap.set(cv2.CAP_PROP_GAIN, new_settings.gain / 100.0)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, new_settings.exposure / 100.0)
        self.cap.set(cv2.CAP_PROP_FOCUS, new_settings.focus / 100.0)
        
        # Update the settings object
        self.settings = new_settings
        
        # Restart processing/recording if needed
        if self.settings.enable_processing and not self.processing_thread:
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
        
        if self.settings.enable_recording and not self.recording_thread:
            self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
            self.recording_thread.start()


class CameraControlGUI:
    """GUI for controlling camera settings."""
    
    def __init__(self, root: tk.Tk, camera_capture: CameraCapture):
        self.root = root
        self.camera_capture = camera_capture
        self.settings = camera_capture.settings
        self.window = tk.Toplevel(root)
        self.window.title(f"Camera {self.settings.camera_id} Controls")
        self.window.geometry("400x600")
        
        # Create UI elements
        self._create_widgets()
        
        # Update UI periodically
        self.window.after(100, self._update_ui)
    
    def _create_widgets(self):
        """Create all UI widgets."""
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Camera info
        info_frame = ttk.LabelFrame(main_frame, text="Camera Info", padding="5")
        info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(info_frame, text=f"Camera ID: {self.settings.camera_id}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Name: {self.settings.name}").pack(anchor=tk.W)
        
        # Display settings
        display_frame = ttk.LabelFrame(main_frame, text="Display Settings", padding="5")
        display_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(display_frame, text="Scale:").pack(anchor=tk.W)
        self.scale_slider = ttk.Scale(
            display_frame, 
            from_=0.1, 
            to=1.0, 
            value=self.settings.display_scale,
            command=self._on_scale_change
        )
        self.scale_slider.pack(fill=tk.X)
        
        # Image settings
        image_frame = ttk.LabelFrame(main_frame, text="Image Settings", padding="5")
        image_frame.pack(fill=tk.X, pady=5)
        
        self.brightness_slider = self._create_slider(
            image_frame, "Brightness", 
            self.settings.brightness, 
            self._on_brightness_change
        )
        
        self.contrast_slider = self._create_slider(
            image_frame, "Contrast", 
            self.settings.contrast, 
            self._on_contrast_change
        )
        
        self.saturation_slider = self._create_slider(
            image_frame, "Saturation", 
            self.settings.saturation, 
            self._on_saturation_change
        )
        
        self.hue_slider = self._create_slider(
            image_frame, "Hue", 
            self.settings.hue, 
            self._on_hue_change
        )
        
        # Processing settings
        processing_frame = ttk.LabelFrame(main_frame, text="Processing", padding="5")
        processing_frame.pack(fill=tk.X, pady=5)
        
        self.enable_processing_var = tk.BooleanVar(value=self.settings.enable_processing)
        processing_check = ttk.Checkbutton(
            processing_frame, 
            text="Enable Processing", 
            variable=self.enable_processing_var,
            command=self._on_processing_toggle
        )
        processing_check.pack(anchor=tk.W)
        
        ttk.Label(processing_frame, text="Algorithm:").pack(anchor=tk.W)
        self.algorithm_combo = ttk.Combobox(
            processing_frame, 
            values=["edge_detection", "blur", "grayscale"],
            state="readonly"
        )
        self.algorithm_combo.set(self.settings.processing_algorithm)
        self.algorithm_combo.pack(fill=tk.X)
        self.algorithm_combo.bind("<<ComboboxSelected>>", self._on_algorithm_change)
        
        # Recording settings
        recording_frame = ttk.LabelFrame(main_frame, text="Recording", padding="5")
        recording_frame.pack(fill=tk.X, pady=5)
        
        self.enable_recording_var = tk.BooleanVar(value=self.settings.enable_recording)
        recording_check = ttk.Checkbutton(
            recording_frame, 
            text="Enable Recording", 
            variable=self.enable_recording_var,
            command=self._on_recording_toggle
        )
        recording_check.pack(anchor=tk.W)
        
        ttk.Label(recording_frame, text="Output Directory:").pack(anchor=tk.W)
        dir_frame = ttk.Frame(recording_frame)
        dir_frame.pack(fill=tk.X)
        
        self.output_dir_var = tk.StringVar(value=self.settings.output_directory)
        output_dir_entry = ttk.Entry(dir_frame, textvariable=self.output_dir_var)
        output_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(dir_frame, text="Browse", command=self._browse_output_dir)
        browse_btn.pack(side=tk.RIGHT, padx=5)
        
        ttk.Label(recording_frame, text="Format:").pack(anchor=tk.W)
        self.format_combo = ttk.Combobox(
            recording_frame, 
            values=["mp4", "avi"],
            state="readonly"
        )
        self.format_combo.set(self.settings.recording_format)
        self.format_combo.pack(fill=tk.X)
        self.format_combo.bind("<<ComboboxSelected>>", self._on_format_change)
        
        # Status
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(anchor=tk.W)
        
        # Close button
        close_btn = ttk.Button(main_frame, text="Close", command=self.window.destroy)
        close_btn.pack(pady=10)
    
    def _create_slider(self, parent: tk.Widget, label: str, value: float, command) -> ttk.Scale:
        """Helper to create a slider with label."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(frame, text=label).pack(side=tk.LEFT)
        
        slider = ttk.Scale(
            frame, 
            from_=0, 
            to=100, 
            value=value,
            command=command
        )
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        return slider
    
    def _update_ui(self):
        """Update UI elements periodically."""
        # Update status
        if self.camera_capture.is_running:
            status_text = f"Running | Frames: {self.camera_capture.frame_count}"
            if self.camera_capture.last_frame_time:
                elapsed = time.time() - self.camera_capture.last_frame_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                status_text += f" | FPS: {fps:.1f}"
        else:
            status_text = "Stopped"
        
        self.status_label.config(text=status_text)
        
        # Schedule next update
        self.window.after(100, self._update_ui)
    
    def _on_scale_change(self, value):
        """Handle display scale change."""
        self.settings.display_scale = float(value)
    
    def _on_brightness_change(self, value):
        """Handle brightness change."""
        self.settings.brightness = float(value)
        self.camera_capture.update_settings(self.settings)
    
    def _on_contrast_change(self, value):
        """Handle contrast change."""
        self.settings.contrast = float(value)
        self.camera_capture.update_settings(self.settings)
    
    def _on_saturation_change(self, value):
        """Handle saturation change."""
        self.settings.saturation = float(value)
        self.camera_capture.update_settings(self.settings)
    
    def _on_hue_change(self, value):
        """Handle hue change."""
        self.settings.hue = float(value)
        self.camera_capture.update_settings(self.settings)
    
    def _on_processing_toggle(self):
        """Handle processing toggle."""
        self.settings.enable_processing = self.enable_processing_var.get()
        self.camera_capture.update_settings(self.settings)
    
    def _on_algorithm_change(self, event):
        """Handle algorithm selection change."""
        self.settings.processing_algorithm = self.algorithm_combo.get()
        self.camera_capture.update_settings(self.settings)
    
    def _on_recording_toggle(self):
        """Handle recording toggle."""
        self.settings.enable_recording = self.enable_recording_var.get()
        self.settings.output_directory = self.output_dir_var.get()
        self.camera_capture.update_settings(self.settings)
    
    def _on_format_change(self, event):
        """Handle format selection change."""
        self.settings.recording_format = self.format_combo.get()
        self.camera_capture.update_settings(self.settings)
    
    def _browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)
            self.settings.output_directory = directory


class MultiCameraApp:
    """Main application for multi-camera capture and display."""
    
    def __init__(self):
        self.cameras: Dict[int, CameraCapture] = {}
        self.control_guis: Dict[int, CameraControlGUI] = {}
        self.root = None
        self.display_window_name = "Multi-Camera View"
        self.is_running = False
        
    def initialize(self, camera_ids: List[int]):
        """Initialize cameras."""
        for cam_id in camera_ids:
            settings = CameraSettings(cam_id, f"Camera {cam_id}")
            camera = CameraCapture(cam_id, settings)
            
            if camera.initialize():
                self.cameras[cam_id] = camera
                logger.info(f"Initialized camera {cam_id}")
            else:
                logger.warning(f"Failed to initialize camera {cam_id}")
    
    def start(self):
        """Start all cameras."""
        for cam_id, camera in self.cameras.items():
            camera.start()
        self.is_running = True
        logger.info("All cameras started")
    
    def stop(self):
        """Stop all cameras."""
        self.is_running = False
        for cam_id, camera in self.cameras.items():
            camera.stop()
        logger.info("All cameras stopped")
    
    def create_gui(self):
        """Create the main GUI."""
        self.root = tk.Tk()
        self.root.title("Multi-Camera Control")
        self.root.geometry("300x200")
        
        # Create main control panel
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Start/Stop buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        start_btn = ttk.Button(button_frame, text="Start All", command=self._start_all)
        start_btn.pack(side=tk.LEFT, padx=5)
        
        stop_btn = ttk.Button(button_frame, text="Stop All", command=self._stop_all)
        stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Camera controls
        camera_frame = ttk.LabelFrame(main_frame, text="Camera Controls", padding="5")
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        for cam_id in self.cameras.keys():
            btn = ttk.Button(
                camera_frame, 
                text=f"Camera {cam_id} Settings",
                command=lambda cid=cam_id: self._show_camera_controls(cid)
            )
            btn.pack(fill=tk.X, pady=2)
        
        # Display button
        display_btn = ttk.Button(main_frame, text="Show Display", command=self._show_display)
        display_btn.pack(pady=10)
        
        # Quit button
        quit_btn = ttk.Button(main_frame, text="Quit", command=self._quit)
        quit_btn.pack(pady=5)
        
        # Start the display loop
        self.root.after(100, self._update_display)
    
    def _start_all(self):
        """Start all cameras."""
        self.start()
    
    def _stop_all(self):
        """Stop all cameras."""
        self.stop()
    
    def _show_camera_controls(self, camera_id: int):
        """Show controls for a specific camera."""
        if camera_id in self.control_guis:
            # Bring existing window to front
            self.control_guis[camera_id].window.lift()
        else:
            # Create new control GUI
            gui = CameraControlGUI(self.root, self.cameras[camera_id])
            self.control_guis[camera_id] = gui
    
    def _show_display(self):
        """Show the display window."""
        self._update_display()
    
    def _update_display(self):
        """Update the display window with frames from all cameras."""
        if not self.is_running:
            self.root.after(100, self._update_display)
            return
        
        # Calculate display layout
        num_cameras = len(self.cameras)
        if num_cameras == 0:
            self.root.after(100, self._update_display)
            return
        
        # Determine grid layout
        cols = int(np.ceil(np.sqrt(num_cameras)))
        rows = int(np.ceil(num_cameras / cols))
        
        # Create a blank canvas
        max_width = 0
        max_height = 0
        
        # Get frames from all cameras
        frames = []
        for cam_id, camera in self.cameras.items():
            frame_data = camera.get_frame()
            if frame_data:
                frame, timestamp = frame_data
                # Scale frame
                scale = camera.settings.display_scale
                if scale != 1.0:
                    width = int(frame.shape[1] * scale)
                    height = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (width, height))
                
                frames.append((cam_id, frame))
                max_width = max(max_width, frame.shape[1])
                max_height = max(max_height, frame.shape[0])
        
        if not frames:
            self.root.after(100, self._update_display)
            return
        
        # Create a grid to hold all frames
        grid_height = max_height * rows
        grid_width = max_width * cols
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Place each frame in the grid
        for i, (cam_id, frame) in enumerate(frames):
            row = i // cols
            col = i % cols
            
            x_offset = col * max_width
            y_offset = row * max_height
            
            # Ensure frame fits in the grid cell
            if frame.shape[0] <= max_height and frame.shape[1] <= max_width:
                grid[y_offset:y_offset+frame.shape[0], x_offset:x_offset+frame.shape[1]] = frame
            else:
                # Scale down if needed
                scaled_frame = cv2.resize(frame, (max_width, max_height))
                grid[y_offset:y_offset+max_height, x_offset:x_offset+max_width] = scaled_frame
            
            # Add camera ID label
            cv2.putText(
                grid, 
                f"Camera {cam_id}", 
                (x_offset + 10, y_offset + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
        
        # Display the grid
        cv2.imshow(self.display_window_name, grid)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            self._quit()
        
        # Schedule next update
        self.root.after(30, self._update_display)
    
    def _quit(self):
        """Quit the application."""
        self.stop()
        cv2.destroyAllWindows()
        if self.root:
            self.root.quit()
        logger.info("Application closed")
    
    def run(self):
        """Run the main application loop."""
        self.create_gui()
        self.root.mainloop()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Camera Live Capture and Display"
    )
    
    parser.add_argument(
        '--cameras', 
        type=int, 
        nargs='+', 
        default=[0],
        help="List of camera IDs to use (default: [0])"
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help="Configuration file to load camera settings"
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Check if we're running on a system with GUI support
    if platform.system() == 'Linux' and not os.environ.get('DISPLAY'):
        logger.error("No display found. Make sure you have X11 forwarding enabled.")
        return
    
    # Initialize the application
    app = MultiCameraApp()
    
    # Initialize cameras
    app.initialize(args.cameras)
    
    if not app.cameras:
        logger.error("No cameras available. Exiting.")
        return
    
    # Start the application
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        app.stop()


if __name__ == "__main__":
    main()