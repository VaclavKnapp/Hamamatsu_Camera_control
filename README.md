# Hamamatsu ORCA-Quest qCMOS Camera C15550-20UP Control Code

![Image_1](https://hub.hamamatsu.com/us/en/ask-an-engineer/imaging/orca-quest-questions-and-answers/_jcr_content/root/container/container_1049569944/container/image.coreimg.jpeg/1732078973730/fig01-orca.jpeg)

## Overview

This Python code provides a comprehensive interface for controlling the Hamamatsu ORCA-Quest qCMOS Camera model C15550-20UP. It leverages the `pydase` library to create a data service with a web-based frontend for interactive control, allowing users to configure camera settings, define regions of interest (ROIs), acquire frames, compute photoelectron counts, and log measurements to HDF5 files. The camera is accessed through the `pylablib` library's DCAM interface, which wraps Hamamatsu's DCAM-API for scientific cameras.

The code operates in a stateful manner, preserving ROI configurations via JSON files and logging data for post-analysis. It includes threading for asynchronous acquisition to prevent blocking the main process.

## Dependencies and Setup

The code requires the following Python libraries:
- `base64`: For encoding images to base64 strings for display.
- `numpy` (np): For numerical operations, including array manipulations and statistical computations (e.g., sum and mean for photoelectron counts).
- `time`: For timing frame rates and display updates.
- `threading`: For running the acquisition loop in a background thread.
- `cv2` (OpenCV): For image processing, such as drawing ROIs, resizing frames, and encoding to JPEG.
- `pydase`: Core framework for creating data services with web frontends; used for properties, methods, and the `Image` component for live viewing.
- `weakref`: For weak references to parent objects in ROI classes to avoid circular references.
- `typing` (List): For type hints.
- `logging`: For setting log levels (e.g., suppressing pydase warnings).
- `h5py`: For creating and writing to HDF5 files for measurement logging.
- `json`: For saving and loading ROI configurations.
- `os`: For file existence checks.
- `pylablib` (pll): Provides the DCAM camera interface (`DCAM.DCAMCamera`).
- `gc`: For manual garbage collection to manage memory during long acquisitions.

**Installation Notes:**
- Install via pip: `pip install numpy opencv-python pydase pylablib h5py`.
- The DCAM-API must be installed separately from Hamamatsu's software media or website, as it provides the driver for camera communication. Ensure the camera is connected via CoaXPress or USB and recognized by the system (e.g., via DCAM-API tools).
- API for windows can be installed through this link: 'https://www.hamamatsu.com/eu/en/product/cameras/software/driver-software/dcam-api-for-windows.html'
- No additional packages can be installed at runtime due to the code interpreter environment constraints.
- Run the script in a Python 3.12+ environment.

**Hardware Requirements:**
- The camera must be connected and powered on. The code assumes the camera is at index 0 (`DCAM.DCAMCamera(idx=0)`).
- For external triggering, use the SMA connector with TTL or 3.3 V LVCMOS signals, as specified in the manual.

## Class Descriptions and Functions

The code defines two main classes: `ROI` and `PhotoelectronCamera`. Both inherit from `pydase.DataService`, enabling them to be exposed via a web interface for real-time interaction.

### ROI Class

This class represents a Region of Interest on the camera sensor. It allows defining rectangular areas for focused photoelectron analysis, which is useful for isolating specific features in the image while ignoring irrelevant areas, reducing computational load.

#### Initialization
- `__init__(self, parent, name: str, x: int = 0, y: int = 0, width: int = 100, height: int = 100, enabled: bool = True)`:
  - `parent`: Weak reference to the `PhotoelectronCamera` instance to allow saving ROIs without circular references.
  - `name`: Identifier for the ROI (e.g., "ROI1").
  - `x`, `y`: Top-left coordinates (in pixels, relative to full sensor).
  - `width`, `height`: Dimensions in pixels.
  - `enabled`: Boolean to toggle ROI computation and display.
  - Reasoning: Coordinates must align with sensor constraints (e.g., multiples of 4 for subarray compatibility, as per manual's subarray mode requiring 4-pixel steps to match hardware readout architecture).

#### Properties
- `x`, `y`, `width`, `height`, `enabled`: Getters and setters that update values and trigger saving ROIs to JSON. These use Python's `@property` decorator for controlled access.
- `photoelectron_count`: Rounded total photoelectrons in the ROI (computed during acquisition).
- `photoelectron_counts_pp`: Rounded mean photoelectrons per pixel in the ROI.

#### Methods
- `delete(self)`: Removes the ROI from the parent's list and saves changes. Decorated with `@frontend` for web exposure.
  - Reasoning: Allows dynamic ROI management via the web interface, ensuring persistence via JSON.

Internal variables like `_total_pe` and `_mean_pe_per_pixel` store raw computations before rounding.

### PhotoelectronCamera Class

This is the main class for camera control. It manages acquisition, settings, ROIs, and logging.

#### Initialization
- `__init__(self, frames_per_chunk=20)`:
  - `frames_per_chunk`: Number of frames per acquisition sequence (default 20) to balance performance and memory.
  - Initializes properties like exposure time (default 0.2 s), scan mode ("UltraQuiet"), crop percentages (0%), and loads ROIs from "rois.json" if present.
  - Creates an `Image` object for live view.
  - Reasoning: Default to UltraQuiet for low-noise applications, as per manual's emphasis on 0.27 e- rms noise for photon resolving. Frames per chunk optimizes sequence mode in DCAM-API to avoid timeouts.

#### Properties
- `exposure_time`: Getter/setter for exposure (float in seconds). Updates camera attribute if running.
  - Range (from manual): 7.2 Î¼s to 1800 s in Standard mode; 199.9 ms to 1800 s in Ultra Quiet (internal trigger). Reasoning: Shorter exposures in Standard for high-speed imaging; longer minima in Ultra Quiet to allow slow readout for noise reduction.
- `photoelectron_counts`: Total photoelectrons in the frame.
- `photoelectron_counts_pp`: Mean photoelectrons per pixel.
- `FPS`: Frames per second, computed in real-time.
- `external_trigger`: Boolean to enable external triggering. Resets/creates HDF5 files and restarts camera if changed.
  - Reasoning: External trigger synchronizes with other instruments (e.g., lasers), using SMA input with positive polarity for edge triggering, as per manual for precise timing in experiments.
- `top_crop_percent`, `bottom_crop_percent`: Percentages to crop vertically (0-100). Restarts camera to apply subarray.
  - Reasoning: Crops reduce readout area for faster FPS, aligning with manual's subarray mode to focus on relevant sensor portions.

- `rois`: List of ROI objects.

#### Methods (Frontend-Exposed)
- `add_roi(self)`: Creates a new ROI with incremental name (e.g., "ROI1") and saves.
- `save_rois(self)`: Serializes ROIs to "rois.json".
- `load_rois(self)`: Loads from "rois.json", handling errors.
- `start_camera(self)`: Starts acquisition thread if not running.
- `stop_camera(self)`: Stops running, logging, and closes camera.
- `start_measurement(self)`: Enables logging to HDF5 files.
- `stop_measurement(self)`: Disables logging and closes HDF5 files.

#### Private Methods
- `_set_subarray(self)`: Configures vertical subarray cropping. Calculates positions/sizes in multiples of 4 (step=4), as per manual (subarray in 4-pixel/line steps to match sensor readout circuitry). Uses `set_roi` for horizontal/vertical binning=1. Falls back to full if error.
  - Reasoning: Subarray mode increases FPS by reading fewer lines (e.g., for lightsheet or focused imaging), but constraints prevent alignment issues.
- `_acquire_loop(self)`: Main loop in thread.
  - Opens camera, sets scan mode, trigger (internal/external with modes: SOURCE=1/2, MODE=1, ACTIVE=1, POLARITY=2), exposure.
  - Gets coeff (0.107 e-/count) and offset (0) for conversion; manual specifies 0.107 as typical sensitivity.
  - Sets up subarray, acquisition (sequence mode).
  - Loop: Waits for frame, reads newest, computes FPS, converts to photoelectrons: `pe = max(0, (frame - offset) * coeff)`.
    - Reasoning: Conversion enables quantitative photon counting, clipping negatives for physical accuracy. Manual notes this for electron-multiplying equivalent in qCMOS.
  - Updates ROIs: Slices pe array, computes sum/mean (adjusted for subarray offsets).
  - Logs to HDF5 if enabled: Resizes datasets dynamically for frame_index, pe_count, pe_pp (full and per ROI).
    - Reasoning: HDF5 for efficient storage of large datasets; resizable for indefinite measurements.
  - Displays every 1s: Normalizes pe to 8-bit, draws ROI rectangles/labels, resizes to 25%, encodes JPEG base64 for web view.
    - Reasoning: Downsampling reduces bandwidth; drawing aids visual ROI adjustment.
  - Handles timeouts/errors, cleans up on stop.
  - Garbage collection every 10 frames to manage memory.

## How to Control the Camera

1. **Run the Script:**
   - Execute `python live_trigger.py`. Starts pydase server at http://localhost:8000.
   - Access via browser for interactive controls: Set properties (e.g., exposure_time=0.1), add/delete ROIs, start/stop camera/measurement.

2. **Basic Operation:**
   - Start camera: Begins live acquisition and view update.
   - Add ROIs: Define areas; enable/disable for selective computation.
   - Exposure: Adjust for light levels; shorter for bright scenes to avoid saturation.
   - Cropping: Use top/bottom percent to focus vertically, increasing FPS.

3. **External Triggering:**
   - Set `external_trigger=True`: Configures camera for external edge trigger (positive polarity).
   - Connect signal to SMA; camera waits for triggers.
   - Starts logging per frame to "full_frame.h5" and "roi_*.h5" (includes ROI metadata).

4. **Measurement Logging:**
   - Start measurement: Logs quantitative data for analysis (e.g., time-series photoelectron counts).
   - Stop: Closes files safely.

5. **Troubleshooting:**
   - Errors in setting attributes: Check manual for supported values (e.g., exposure range).
   - Memory issues: Acquisition uses float32 arrays; gc.collect() helps.
   - Cleanup: Always stop_camera() on exit to release resources.
