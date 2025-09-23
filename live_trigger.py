import base64
import numpy as np
import time
import threading
import cv2
import pydase
import weakref
from typing import List
from pydase.components import Image
from pydase.utils.decorators import frontend
import logging
import h5py
import json
import os
import pylablib as pll
from pylablib.devices import DCAM
from pylablib.devices.DCAM import DCAMTimeoutError
import gc

logging.getLogger('pydase').setLevel(logging.WARNING)

class ROI(pydase.DataService):
    def __init__(self, parent, name: str, x: int = 0, y: int = 0, width: int = 100, height: int = 100, enabled: bool = True):
        super().__init__()
        self._parent = weakref.ref(parent)
        self.name = name
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._enabled = enabled
        self._total_pe = 0.0
        self._mean_pe_per_pixel = 0.0

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        parent = self._parent()
        if parent is not None:
            parent.save_rois()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
        parent = self._parent()
        if parent is not None:
            parent.save_rois()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        parent = self._parent()
        if parent is not None:
            parent.save_rois()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value
        parent = self._parent()
        if parent is not None:
            parent.save_rois()

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value
        parent = self._parent()
        if parent is not None:
            parent.save_rois()

    @frontend
    def delete(self):
        parent = self._parent()
        if parent is not None:
            parent.rois.remove(self)
            parent.save_rois()

    @property
    def photoelectron_count(self):
        return round(float(self._total_pe), 2)

    @property
    def photoelectron_counts_pp(self):
        return round(float(self._mean_pe_per_pixel), 2)

class PhotoelectronCamera(pydase.DataService):
    def __init__(self, frames_per_chunk=20):
        super().__init__()
        self._camera = None
        self._exposure_time = 0.2
        self._frames_per_chunk = frames_per_chunk
        self.Camera_view = Image()
        self._p_e = 0.0
        self._p_e_p = 0.0
        self._fps = 0.0
        self.rois: List[ROI] = []
        self._running = False
        self._logging = False  # For measurement logging
        self._external_trigger = False
        self._frame_count = 0
        self._h5_full = None
        self._h5_rois = {}
        self._photoelectrons = None
        self._display_frame = None
        self._display_small = None
        self._top_crop_percent = 0.0
        self._bottom_crop_percent = 0.0
        self._scan_mode = "UltraQuiet"  # Default; options: "Standard", "UltraQuiet"
        if os.path.exists("rois.json"):
            self.load_rois()

    @property
    def exposure_time(self):
        return self._exposure_time

    @property
    def photoelectron_counts(self):
        return round(float(self._p_e), 2)

    @property
    def photoelectron_counts_pp(self):
        return round(float(self._p_e_p), 2)

    @property
    def FPS(self):
        return round(float(self._fps), 2)

    @property
    def external_trigger(self):
        return self._external_trigger

    @property
    def top_crop_percent(self):
        return self._top_crop_percent

    @property
    def bottom_crop_percent(self):
        return self._bottom_crop_percent

    @property
    def scan_mode(self):
        return self._scan_mode

    @exposure_time.setter
    def exposure_time(self, value):
        self._exposure_time = float(value)
        if self._running and self._camera is not None:
            self._camera.set_attribute_value("EXPOSURE_TIME", self._exposure_time)

    @external_trigger.setter
    def external_trigger(self, value):
        self._external_trigger = bool(value)
        if value:
            # Reset and create .h5 files with resizable datasets
            with h5py.File("full_frame.h5", "w") as f:
                f.create_dataset("frame_index", shape=(0,), maxshape=(None,), dtype=int, chunks=(1024,))
                f.create_dataset("photoelectron_count", shape=(0,), maxshape=(None,), dtype=float, chunks=(1024,))
                f.create_dataset("photoelectron_counts_pp", shape=(0,), maxshape=(None,), dtype=float, chunks=(1024,))
            for roi in self.rois:
                with h5py.File(f"roi_{roi.name}.h5", "w") as f:
                    f.create_dataset("frame_index", shape=(0,), maxshape=(None,), dtype=int, chunks=(1024,))
                    f.create_dataset("photoelectron_count", shape=(0,), maxshape=(None,), dtype=float, chunks=(1024,))
                    f.create_dataset("photoelectron_counts_pp", shape=(0,), maxshape=(None,), dtype=float, chunks=(1024,))
                    f.create_dataset("x", data=roi.x)
                    f.create_dataset("y", data=roi.y)
                    f.create_dataset("width", data=roi.width)
                    f.create_dataset("height", data=roi.height)
            self._frame_count = 0
            self.start_measurement()
        else:
            self.stop_measurement()
        if self._running:
            self.stop_camera()
            self.start_camera()

    @top_crop_percent.setter
    def top_crop_percent(self, value):
        self._top_crop_percent = float(value)
        if self._running:
            self.stop_camera()
            self.start_camera()

    @bottom_crop_percent.setter
    def bottom_crop_percent(self, value):
        self._bottom_crop_percent = float(value)
        if self._running:
            self.stop_camera()
            self.start_camera()

    @scan_mode.setter
    def scan_mode(self, value):
        if value not in ["Standard", "UltraQuiet"]:
            raise ValueError("Scan mode must be 'Standard' or 'UltraQuiet'")
        self._scan_mode = value
        if self._running:
            self.stop_camera()
            self.start_camera()

    @frontend
    def add_roi(self):
        num = len(self.rois) + 1
        name = f"ROI{num}"
        new_roi = ROI(parent=self, name=name)
        self.rois.append(new_roi)
        self.save_rois()

    @frontend
    def save_rois(self):
        rois_data = [
            {
                "name": roi.name,
                "x": roi.x,
                "y": roi.y,
                "width": roi.width,
                "height": roi.height,
                "enabled": roi.enabled
            }
            for roi in self.rois
        ]
        with open("rois.json", "w") as f:
            json.dump(rois_data, f)

    @frontend
    def load_rois(self):
        try:
            with open("rois.json", "r") as f:
                rois_data = json.load(f)
            self.rois = []
            for data in rois_data:
                self.rois.append(ROI(parent=self, **data))
        except Exception as e:
            print(f"Error loading ROIs: {e}")

    @frontend
    def start_camera(self):
        if self._running:
            return
        self._running = True
        self._acquisition_thread = threading.Thread(target=self._acquire_loop, daemon=True)
        self._acquisition_thread.start()

    @frontend
    def stop_camera(self):
        self._running = False
        self._logging = False
        if self._camera is not None:
            try:
                self._camera.stop_acquisition()
            except Exception as e:
                print(f"Error stopping acquisition: {e}")
        if hasattr(self, '_acquisition_thread') and self._acquisition_thread.is_alive():
            self._acquisition_thread.join(timeout=5.0)

    @frontend
    def start_measurement(self):
        self._logging = True
        self._h5_full = h5py.File("full_frame.h5", "a")
        for roi in self.rois:
            self._h5_rois[roi.name] = h5py.File(f"roi_{roi.name}.h5", "a")

    @frontend
    def stop_measurement(self):
        self._logging = False
        if self._h5_full:
            self._h5_full.close()
            self._h5_full = None
        for h5 in list(self._h5_rois.values()):
            h5.close()
        self._h5_rois = {}

    def _set_subarray(self):
        full_width, full_height = self._camera.get_detector_size()
        step = 4  # From camera specifications
        top_lines = int(full_height * self._top_crop_percent / 100 // step) * step
        bottom_lines = int(full_height * self._bottom_crop_percent / 100 // step) * step
        vsize = full_height - top_lines - bottom_lines
        vsize = max(step, (vsize // step) * step)
        vpos = top_lines
        hpos = 0
        hsize = full_width

        # Use pylablib's set_roi method for proper handling
        try:
            self._camera.set_roi(hstart=hpos, hend=hpos + hsize, vstart=vpos, vend=vpos + vsize, hbin=1, vbin=1)
        except Exception as e:
            print(f"Error setting ROI: {e}")
            # Fallback to full if error
            self._camera.set_roi()
            vpos = 0
            vsize = full_height
            hpos = 0
            hsize = full_width

        subarray_on = vsize < full_height
        return subarray_on, hpos, hsize, vpos, vsize

    def _acquire_loop(self):
        try:
            self._camera = DCAM.DCAMCamera(idx=0)
            # Set scan mode
            scan_value = 1 if self._scan_mode == "Standard" else 2  # Assumed values; verify via get_attributes() if needed
            try:
                self._camera.set_attribute_value("SCAN_MODE", scan_value)
            except Exception as e:
                print(f"Error setting SCAN_MODE: {e}. Using default.")

            if self._external_trigger:
                self._camera.set_attribute_value("TRIGGER_SOURCE", 2.0)  # 'EXTERNAL'
                self._camera.set_attribute_value("TRIGGER_MODE", 1.0)  # 'NORMAL'
                self._camera.set_attribute_value("TRIGGER_ACTIVE", 1.0)  # 'EDGE'
                self._camera.set_attribute_value("TRIGGER_POLARITY", 2.0)  # 'POSITIVE'
            else:
                self._camera.set_attribute_value("TRIGGER_SOURCE", 1.0)  # 'INTERNAL'
            self._camera.set_attribute_value("EXPOSURE_TIME", self._exposure_time)

            try:
                coeff = self._camera.get_attribute_value("CONVERSION_FACTOR_COEFF")
                offset = self._camera.get_attribute_value("CONVERSION_FACTOR_OFFSET")
            except Exception as e:
                print(f"Error getting conversion factors: {e}. Using default values.")
                coeff = 0.107
                offset = 0

            subarray_on, hpos, hsize, vpos, vsize = self._set_subarray()

            self._camera.setup_acquisition(mode="sequence", nframes=self._frames_per_chunk)
            self._camera.start_acquisition()

            full_width, full_height = self._camera.get_detector_size()
            prev_time = time.time()
            last_display_time = 0.0

            self._photoelectrons = np.empty((vsize, hsize), dtype=np.float32)
            temp_display = np.empty((vsize, hsize), dtype=np.uint8)
            self._display_frame = np.empty((full_height, full_width), dtype=np.uint8)
            new_height = int(full_height * 0.25)
            new_width = int(full_width * 0.25)
            self._display_small = np.empty((new_height, new_width), dtype=np.uint8)

            while self._running:
                try:
                    got_frame = self._camera.wait_for_frame(timeout=0.1, error_on_stopped=False)
                    if not got_frame:
                        continue
                    frame = self._camera.read_newest_image()
                    if frame is None:
                        continue

                    current_time = time.time()
                    self._fps = 1 / (current_time - prev_time) if self._frame_count > 0 else 0.0
                    prev_time = current_time

                    self._photoelectrons[:] = frame.astype(np.float32)
                    self._photoelectrons -= offset
                    self._photoelectrons *= coeff
                    np.maximum(self._photoelectrons, 0, out=self._photoelectrons)

                    self._p_e = float(np.sum(self._photoelectrons))
                    self._p_e_p = float(np.mean(self._photoelectrons))

                    for roi in self.rois:
                        if roi.enabled:
                            roi_x, roi_y, roi_w, roi_h = roi.x, roi.y, roi.width, roi.height
                            rel_x = roi_x - hpos
                            rel_y = roi_y - vpos
                            start_x = max(0, rel_x)
                            end_x = min(hsize, rel_x + roi_w)
                            start_y = max(0, rel_y)
                            end_y = min(vsize, rel_y + roi_h)
                            if end_x > start_x and end_y > start_y:
                                roi_slice = self._photoelectrons[start_y:end_y, start_x:end_x]
                                roi._total_pe = float(np.sum(roi_slice))
                                area = (end_x - start_x) * (end_y - start_y)
                                roi._mean_pe_per_pixel = roi._total_pe / area if area > 0 else 0.0
                            else:
                                roi._total_pe = 0.0
                                roi._mean_pe_per_pixel = 0.0

                    self._frame_count += 1

                    if self._logging:
                        for key in ["frame_index", "photoelectron_count", "photoelectron_counts_pp"]:
                            ds = self._h5_full[key]
                            ds.resize((ds.shape[0] + 1,))
                        self._h5_full["frame_index"][-1] = self._frame_count
                        self._h5_full["photoelectron_count"][-1] = self._p_e
                        self._h5_full["photoelectron_counts_pp"][-1] = self._p_e_p

                        for roi in self.rois:
                            if roi.enabled:
                                h5 = self._h5_rois[roi.name]
                                for key in ["frame_index", "photoelectron_count", "photoelectron_counts_pp"]:
                                    ds = h5[key]
                                    ds.resize((ds.shape[0] + 1,))
                                h5["frame_index"][-1] = self._frame_count
                                h5["photoelectron_count"][-1] = roi._total_pe
                                h5["photoelectron_counts_pp"][-1] = roi._mean_pe_per_pixel

                    if current_time - last_display_time >= 1.0:
                        max_val = self._photoelectrons.max()
                        if max_val > 0:
                            np.divide(self._photoelectrons, max_val, out=self._photoelectrons)
                            np.multiply(self._photoelectrons, 255, out=self._photoelectrons)
                            temp_display[:] = self._photoelectrons.astype(np.uint8)
                        else:
                            temp_display.fill(0)

                        self._display_frame.fill(0)
                        self._display_frame[vpos : vpos + vsize, hpos : hpos + hsize] = temp_display

                        if subarray_on:
                            cv2.line(self._display_frame, (0, vpos), (full_width - 1, vpos), 255, 2)
                            cv2.line(self._display_frame, (0, vpos + vsize), (full_width - 1, vpos + vsize), 255, 2)

                        for i, roi in enumerate(self.rois):
                            if roi.enabled:
                                cv2.rectangle(self._display_frame, (roi.x, roi.y), (roi.x + roi.width, roi.y + roi.height), 255, 2)
                                cv2.putText(self._display_frame, f"{roi.name} ({i+1})", (roi.x, roi.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.75, 255, 2)

                        cv2.resize(self._display_frame, (new_width, new_height), dst=self._display_small, interpolation=cv2.INTER_AREA)
                        ret, buf = cv2.imencode(".jpg", self._display_small, [cv2.IMWRITE_JPEG_QUALITY, 50])
                        if ret:
                            self.Camera_view.load_from_base64(base64.b64encode(buf.tobytes()))
                        else:
                            print("Failed to encode frame.")
                        last_display_time = current_time

                    gc.collect()
                    if self._frame_count % 10 == 0:
                        gc.collect()

                except DCAMTimeoutError:
                    continue
                except Exception as e:
                    print(f"Acquisition error: {e}")
        finally:
            if self._camera is not None:
                try:
                    self._camera.stop_acquisition()
                    self._camera.close()
                except Exception as e:
                    print(f"Error closing camera: {e}")
            self._camera = None

if __name__ == "__main__":
    service_instance = PhotoelectronCamera(frames_per_chunk=20)
    server = pydase.Server(service=service_instance, web_port=8000, generate_web_settings=True)
    try:
        server.run()  # Starts web server; access at http://localhost:8000
    finally:
        service_instance.stop_camera()
