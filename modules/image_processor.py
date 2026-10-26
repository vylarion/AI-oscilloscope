"""
Image Processing Engine — OpenCV pipeline for oscilloscope screenshot analysis.

Pipeline:
    load_image → preprocess → find_plot_area → detect_grid →
    detect_scale (OCR optional) → extract_trace → pixels_to_physical
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GridInfo:
    """Detected grid (graticule) geometry in pixel space."""
    h_lines: list[int]          # y-coords of horizontal grid lines
    v_lines: list[int]          # x-coords of vertical grid lines
    cell_h_px: float            # pixels per vertical division
    cell_w_px: float            # pixels per horizontal division
    plot_x1: int                # left edge of plot area
    plot_x2: int                # right edge of plot area
    plot_y1: int                # top edge of plot area
    plot_y2: int                # bottom edge of plot area
    n_hdiv: int = 8             # number of horizontal (time) divisions
    n_vdiv: int = 10            # number of vertical (voltage) divisions


@dataclass
class ScaleInfo:
    """Physical scale: how much each grid division represents."""
    v_per_div: float = 1.0      # volts per vertical division
    t_per_div: float = 1e-3     # seconds per horizontal division
    v_unit: str = "V"           # "V" or "mV"
    t_unit: str = "ms"          # "s", "ms", "us", "ns"
    source: str = "default"     # "ocr", "user", "default"


@dataclass
class WaveformTrace:
    """Digitized waveform and all context needed for DSP."""
    times: np.ndarray           # seconds, shape (N,)
    voltages: np.ndarray        # volts, shape (N,)
    grid_info: GridInfo
    scale_info: ScaleInfo
    annotated_image: Optional[np.ndarray] = field(default=None, repr=False)
    ocr_texts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Unit parsing helpers
# ---------------------------------------------------------------------------

# Multipliers for voltage/time prefix strings
_V_MULTIPLIERS = {"kv": 1e3, "v": 1.0, "mv": 1e-3, "uv": 1e-6}
_T_MULTIPLIERS = {"ks": 1e3, "s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9, "ps": 1e-12}

_V_PATTERN = re.compile(
    r"([+-]?\d+\.?\d*)\s*(k?m?u?µ?[vV])",
    re.IGNORECASE
)
_T_PATTERN = re.compile(
    r"([+-]?\d+\.?\d*)\s*(k?[mMuUµnNpP]?s)",
    re.IGNORECASE
)
_FREQ_PATTERN = re.compile(
    r"([+-]?\d+\.?\d*)\s*([kKmMgG]?Hz)",
    re.IGNORECASE
)


def _parse_voltage(text: str) -> Optional[float]:
    """Return volts from a string like '500mV', '2.5V', '200uV'."""
    m = _V_PATTERN.search(text)
    if not m:
        return None
    val = float(m.group(1))
    suffix = m.group(2).lower().replace("µ", "u")
    for key, mult in _V_MULTIPLIERS.items():
        if suffix.startswith(key):
            return val * mult
    return val  # assume plain volts


def _parse_time(text: str) -> Optional[float]:
    """Return seconds from a string like '10ms', '500us', '1.5s'."""
    m = _T_PATTERN.search(text)
    if not m:
        return None
    val = float(m.group(1))
    suffix = m.group(2).lower().replace("µ", "u")
    for key, mult in _T_MULTIPLIERS.items():
        if suffix.startswith(key):
            return val * mult
    return val  # assume plain seconds


def _unit_label_voltage(v_per_div: float) -> tuple[float, str]:
    """Return (display_value, unit_string) for a V/div value."""
    if v_per_div >= 1.0:
        return v_per_div, "V"
    elif v_per_div >= 1e-3:
        return v_per_div * 1e3, "mV"
    else:
        return v_per_div * 1e6, "µV"


def _unit_label_time(t_per_div: float) -> tuple[float, str]:
    """Return (display_value, unit_string) for a time/div value."""
    if t_per_div >= 1.0:
        return t_per_div, "s"
    elif t_per_div >= 1e-3:
        return t_per_div * 1e3, "ms"
    elif t_per_div >= 1e-6:
        return t_per_div * 1e6, "µs"
    else:
        return t_per_div * 1e9, "ns"


# ---------------------------------------------------------------------------
# Main processor class
# ---------------------------------------------------------------------------

class ImageProcessor:
    """
    Full pipeline from raw image bytes to a digitized WaveformTrace.

    Usage:
        processor = ImageProcessor()
        trace = processor.process(image_bytes, ocr_reader=reader)
    """

    # Maximum working resolution (rescale larger images)
    MAX_DIM = 1280

    def process(
        self,
        image_bytes: bytes,
        ocr_reader=None,
        user_scale: Optional[ScaleInfo] = None,
    ) -> WaveformTrace:
        """
        Full pipeline: bytes → WaveformTrace.

        Args:
            image_bytes: Raw image file bytes (PNG, JPEG, BMP, etc.)
            ocr_reader:  Optional pre-initialized EasyOCR Reader instance.
            user_scale:  Optional manually-specified ScaleInfo to override OCR/defaults.

        Returns:
            WaveformTrace with digitized (time, voltage) arrays.
        """
        # 1. Load & preprocess
        image = self._load(image_bytes)
        image = self._preprocess(image)

        # 2. Detect grid
        grid = self._detect_grid(image)

        # 3. Determine scale
        ocr_texts: list[str] = []
        if user_scale is not None:
            scale = user_scale
            scale.source = "user"
        elif ocr_reader is not None:
            scale, ocr_texts = self._scale_from_ocr(image, ocr_reader)
        else:
            scale = ScaleInfo()
            scale.source = "default"

        # 4. Extract trace pixels
        x_px, y_px = self._extract_trace(image, grid)

        # 5. Convert to physical units
        times, voltages = self._pixels_to_physical(x_px, y_px, grid, scale)

        # 6. Build annotated image for display
        annotated = self._annotate(image.copy(), grid, x_px, y_px)

        return WaveformTrace(
            times=times,
            voltages=voltages,
            grid_info=grid,
            scale_info=scale,
            annotated_image=annotated,
            ocr_texts=ocr_texts,
        )

    # ------------------------------------------------------------------
    # Step 1 — Load
    # ------------------------------------------------------------------

    def _load(self, image_bytes: bytes) -> np.ndarray:
        """Decode bytes → BGR numpy array, resized if very large."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image. Please upload a valid PNG/JPEG file.")

        h, w = img.shape[:2]
        if max(h, w) > self.MAX_DIM:
            scale = self.MAX_DIM / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        return img

    # ------------------------------------------------------------------
    # Step 2 — Preprocess
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Light denoising to reduce JPEG compression artifacts."""
        return cv2.GaussianBlur(image, (3, 3), sigmaX=0.5)

    # ------------------------------------------------------------------
    # Step 3 — Detect grid
    # ------------------------------------------------------------------

    def _detect_grid(self, image: np.ndarray) -> GridInfo:
        """
        Detect the scope graticule (grid) and return the plot-area bounds.

        Strategy:
        1. Scan row/column brightness profiles to find the actual dark display
           region (excludes UI bars at top/bottom which are lighter).
        2. Try HoughLinesP inside that region to count grid divisions.
        3. Fall back to conservative 10x8 uniform grid if line detection fails.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ------------------------------------------------------------------
        # Step A: Find display area by brightness profile scanning.
        # Scope UI bars (top/bottom) are noticeably lighter than the dark
        # waveform display.  We scan row medians to find the dark region.
        # ------------------------------------------------------------------
        row_med = np.median(gray, axis=1)   # median brightness per row
        col_med = np.median(gray, axis=0)   # median brightness per column

        # Threshold below which a row/col is considered "inside the dark display"
        # Use 60% of the overall image median as cutoff
        brightness_cutoff = max(30, float(np.median(gray)) * 0.70)

        dark_rows = np.where(row_med < brightness_cutoff)[0]
        dark_cols = np.where(col_med < brightness_cutoff)[0]

        # Use the contiguous dark band that contains the image centre
        cy, cx = h // 2, w // 2

        def _contiguous_band(indices, centre, total):
            """Return the contiguous run of indices that contains 'centre'."""
            if len(indices) == 0:
                return int(total * 0.10), int(total * 0.90)
            # Find the run containing centre (or the nearest one)
            closest = int(min(indices, key=lambda v: abs(v - centre)))
            lo, hi = closest, closest
            idx_set = set(int(i) for i in indices)
            while lo - 1 in idx_set:
                lo -= 1
            while hi + 1 in idx_set:
                hi += 1
            return lo, hi

        py1, py2 = _contiguous_band(dark_rows, cy, h)
        px1, px2 = _contiguous_band(dark_cols, cx, w)

        # Safety: require at least 50% of image in each direction
        min_h = int(h * 0.40)
        min_w = int(w * 0.40)
        if py2 - py1 < min_h:
            # Brightness profile didn't discriminate well — use conservative frac
            py1 = int(h * 0.12)
            py2 = int(h * 0.88)
        if px2 - px1 < min_w:
            px1 = int(w * 0.05)
            px2 = int(w * 0.95)

        # Add a small inner margin to stay away from scope border pixels
        margin_y = max(4, int((py2 - py1) * 0.03))
        margin_x = max(4, int((px2 - px1) * 0.02))
        py1 += margin_y
        py2 -= margin_y
        px1 += margin_x
        px2 -= margin_x

        roi = gray[py1:py2, px1:px2]

        # ------------------------------------------------------------------
        # Step B: Hough line detection to count grid divisions.
        # ------------------------------------------------------------------
        edges = cv2.Canny(roi, threshold1=20, threshold2=60)
        min_len = int(min(roi.shape) * 0.4)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=int(min_len * 0.6),
            minLineLength=min_len,
            maxLineGap=15,
        )

        h_lines: list[int] = []
        v_lines: list[int] = []

        if lines is not None:
            for line in lines:
                pts = line.flatten()
                x1_, y1_, x2_, y2_ = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
                dx = abs(x2_ - x1_)
                dy = abs(y2_ - y1_)
                if dx == 0 and dy == 0:
                    continue
                angle = np.degrees(np.arctan2(dy, dx))
                if angle < 10:      # horizontal
                    h_lines.append((y1_ + y2_) // 2 + py1)
                elif angle > 80:   # vertical
                    v_lines.append((x1_ + x2_) // 2 + px1)

        h_lines = self._cluster_lines(sorted(set(h_lines)), min_gap=5)
        v_lines = self._cluster_lines(sorted(set(v_lines)), min_gap=5)

        # Determine division count from detected lines
        n_vdiv = max(len(v_lines) - 1, 1) if len(v_lines) >= 3 else 10
        n_hdiv = max(len(h_lines) - 1, 1) if len(h_lines) >= 3 else 8

        # Only use Hough-derived bounds if lines look plausible
        plot_x1 = int(min(v_lines)) if len(v_lines) >= 2 else px1
        plot_x2 = int(max(v_lines)) if len(v_lines) >= 2 else px2
        plot_y1 = int(min(h_lines)) if len(h_lines) >= 2 else py1
        plot_y2 = int(max(h_lines)) if len(h_lines) >= 2 else py2

        cell_w = (plot_x2 - plot_x1) / max(n_vdiv, 1)
        cell_h = (plot_y2 - plot_y1) / max(n_hdiv, 1)

        # If cells are implausibly small or the bounds extend outside the
        # brightness-detected area, fall back to brightness-detected bounds
        if (
            cell_w < 10 or cell_h < 10
            or plot_x1 < px1 or plot_x2 > px2
            or plot_y1 < py1 or plot_y2 > py2
        ):
            plot_x1, plot_x2 = px1, px2
            plot_y1, plot_y2 = py1, py2
            n_vdiv, n_hdiv = 10, 8
            cell_w = (plot_x2 - plot_x1) / n_vdiv
            cell_h = (plot_y2 - plot_y1) / n_hdiv
            h_lines = [int(plot_y1 + i * cell_h) for i in range(n_hdiv + 1)]
            v_lines = [int(plot_x1 + i * cell_w) for i in range(n_vdiv + 1)]

        logger.debug(
            f"Grid: {n_vdiv}x{n_hdiv} divs, cell {cell_w:.1f}x{cell_h:.1f} px, "
            f"plot area ({plot_x1},{plot_y1})->({plot_x2},{plot_y2})"
        )

        return GridInfo(
            h_lines=h_lines,
            v_lines=v_lines,
            cell_h_px=cell_h,
            cell_w_px=cell_w,
            plot_x1=plot_x1,
            plot_x2=plot_x2,
            plot_y1=plot_y1,
            plot_y2=plot_y2,
            n_hdiv=n_hdiv,
            n_vdiv=n_vdiv,
        )

    @staticmethod
    def _cluster_lines(lines: list[int], min_gap: int = 5) -> list[int]:
        """Merge lines that are closer than min_gap pixels (keep centroid)."""
        if not lines:
            return []
        clusters: list[list[int]] = [[lines[0]]]
        for val in lines[1:]:
            if val - clusters[-1][-1] <= min_gap:
                clusters[-1].append(val)
            else:
                clusters.append([val])
        return [int(np.mean(c)) for c in clusters]

    # ------------------------------------------------------------------
    # Step 4a — Scale from OCR
    # ------------------------------------------------------------------

    def _scale_from_ocr(
        self, image: np.ndarray, ocr_reader
    ) -> tuple[ScaleInfo, list[str]]:
        """
        Run EasyOCR on the image and parse V/div and time/div text.
        Returns a ScaleInfo and a list of all detected text strings.
        """
        try:
            results = ocr_reader.readtext(image, detail=0, paragraph=False)
        except Exception as exc:
            logger.warning(f"EasyOCR failed: {exc}")
            return ScaleInfo(source="default"), []

        texts = [str(r) for r in results]
        full_text = " ".join(texts)
        logger.debug(f"OCR texts: {texts}")

        v_per_div: Optional[float] = None
        t_per_div: Optional[float] = None

        # Try every OCR token for voltage and time
        for token in texts:
            token_clean = token.strip()
            if v_per_div is None:
                v = _parse_voltage(token_clean)
                if v is not None and 1e-4 < v < 1000:
                    v_per_div = v
            if t_per_div is None:
                t = _parse_time(token_clean)
                if t is not None and 1e-10 < t < 100:
                    t_per_div = t

        scale = ScaleInfo(
            v_per_div=v_per_div if v_per_div is not None else 1.0,
            t_per_div=t_per_div if t_per_div is not None else 1e-3,
            source="ocr" if (v_per_div or t_per_div) else "default",
        )

        # Populate unit labels
        _, scale.v_unit = _unit_label_voltage(scale.v_per_div)
        _, scale.t_unit = _unit_label_time(scale.t_per_div)

        logger.info(
            f"Scale from OCR: {scale.v_per_div} V/div, {scale.t_per_div} s/div "
            f"(source={scale.source})"
        )
        return scale, texts

    # ------------------------------------------------------------------
    # Step 5 — Extract trace
    # ------------------------------------------------------------------

    def _extract_trace(
        self, image: np.ndarray, grid: GridInfo
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Isolate the waveform trace within the plot area.

        Strategy:
        1. Add extra inner margin to avoid residual scope UI at plot edges.
        2. Convert ROI to HSV; find the dominant trace hue (most-common
           saturated+bright hue) rather than using all bright pixels.
        3. For each x-column: pick the SINGLE brightest trace pixel
           (argmax) — not the weighted centroid of all bright pixels.
           This is correct because oscilloscope traces are thin lines.
        4. Apply a continuity outlier filter (median-deviation rejection)
           to discard stray pixels that slipped through the color mask.
        5. Smooth the result with Savitzky-Golay.
        """
        # Add inner margin beyond the detected grid boundary
        h_range = grid.plot_y2 - grid.plot_y1
        w_range = grid.plot_x2 - grid.plot_x1
        margin_y = max(6, int(h_range * 0.04))
        margin_x = max(4, int(w_range * 0.02))

        x1 = grid.plot_x1 + margin_x
        x2 = grid.plot_x2 - margin_x
        y1 = grid.plot_y1 + margin_y
        y2 = grid.plot_y2 - margin_y

        if x2 <= x1 or y2 <= y1:
            xs = np.arange(grid.plot_x1, grid.plot_x2 + 1, dtype=float)
            ys = np.full_like(xs, float((grid.plot_y1 + grid.plot_y2) // 2))
            return xs, ys

        roi = image[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0]   # hue   0-179
        s_ch = hsv[:, :, 1]   # saturation 0-255
        v_ch = hsv[:, :, 2]   # value (brightness) 0-255

        # ------------------------------------------------------------------
        # Find the dominant trace hue among saturated+bright pixels.
        # This automatically adapts to any trace color (yellow, green, cyan…)
        # ------------------------------------------------------------------
        candidate_mask = (v_ch > 100) & (s_ch > 60)

        if candidate_mask.sum() > roi_w * 1.5:
            hues = h_ch[candidate_mask]
            hist, _ = np.histogram(hues, bins=36, range=(0, 180))
            dominant_bin = int(np.argmax(hist))
            dominant_hue = dominant_bin * 5  # each bin = 5 degrees

            # Accept pixels within ±20 degrees of dominant hue
            lo = (dominant_hue - 20) % 180
            hi = (dominant_hue + 20) % 180
            if lo <= hi:
                hue_mask = (h_ch >= lo) & (h_ch <= hi)
            else:  # wraps around 0/180
                hue_mask = (h_ch >= lo) | (h_ch <= hi)

            trace_mask = hue_mask & (v_ch > 80) & (s_ch > 40)
        else:
            # Low-saturation display (e.g. white/gray trace on dark background)
            # Fall back to top-percentile brightness
            thresh = max(80, int(np.percentile(v_ch, 88)))
            trace_mask = v_ch > thresh

        # ------------------------------------------------------------------
        # Per-column: find the SINGLE brightest trace pixel (argmax).
        # This works correctly for thin trace lines.
        # ------------------------------------------------------------------
        x_coords: list[int] = []
        y_coords: list[float] = []

        scored = v_ch.astype(float) * trace_mask.astype(float)

        for col in range(roi_w):
            col_score = scored[:, col]
            best_val = col_score.max()
            if best_val < 20:   # no trace pixel found in this column
                continue
            # Sub-pixel centroid within a ±3px window around the peak
            peak_y = int(np.argmax(col_score))
            lo_w = max(0, peak_y - 3)
            hi_w = min(roi_h - 1, peak_y + 3)
            window = col_score[lo_w:hi_w + 1]
            if window.sum() > 0:
                y_ctr = lo_w + float(np.average(np.arange(len(window)), weights=window))
            else:
                y_ctr = float(peak_y)
            x_coords.append(col)
            y_coords.append(y_ctr)

        # ------------------------------------------------------------------
        # Fallback: if color mask caught very little, try brightness only
        # ------------------------------------------------------------------
        if len(x_coords) < roi_w * 0.3:
            x_coords, y_coords = [], []
            thresh = max(80, int(np.percentile(v_ch, 88)))
            for col in range(roi_w):
                col_v = v_ch[:, col].astype(float)
                if col_v.max() < thresh:
                    continue
                peak_y = int(np.argmax(col_v))
                x_coords.append(col)
                y_coords.append(float(peak_y))

        if len(x_coords) < 5:
            logger.warning("Very few trace pixels found; returning flat midline.")
            x_arr = np.arange(roi_w, dtype=float)
            y_arr = np.full(roi_w, roi_h / 2.0)
            return (x_arr + x1), (y_arr + y1)

        x_arr = np.array(x_coords, dtype=float)
        y_arr = np.array(y_coords, dtype=float)

        # Sort by x
        order = np.argsort(x_arr)
        x_arr, y_arr = x_arr[order], y_arr[order]

        # ------------------------------------------------------------------
        # Continuity outlier filter.
        # Points that deviate > 15% of plot height from a local median
        # are almost certainly stray pixels (UI text, grid artefacts).
        # ------------------------------------------------------------------
        if len(y_arr) > 11:
            from scipy.signal import medfilt
            kernel = min(21, (len(y_arr) // 6) * 2 + 1)
            kernel = max(kernel, 3)
            y_med = medfilt(y_arr, kernel_size=kernel)
            deviation = np.abs(y_arr - y_med)
            max_dev = max(5.0, roi_h * 0.15)
            good = deviation < max_dev
            if good.sum() >= 10:
                x_arr = x_arr[good]
                y_arr = y_arr[good]

        # ------------------------------------------------------------------
        # Smooth with Savitzky-Golay
        # ------------------------------------------------------------------
        if len(y_arr) >= 9:
            win = min(31, (len(y_arr) // 6) * 2 + 1)
            win = max(win, 5)
            try:
                y_arr = savgol_filter(y_arr, window_length=win, polyorder=3)
            except Exception:
                pass  # keep unsmoothed if filter fails

        return (x_arr + x1).astype(float), (y_arr + y1).astype(float)

    # ------------------------------------------------------------------
    # Step 6 — Pixels → physical units
    # ------------------------------------------------------------------

    def _pixels_to_physical(
        self,
        x_px: np.ndarray,
        y_px: np.ndarray,
        grid: GridInfo,
        scale: ScaleInfo,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert pixel coordinates to (time_s, voltage_v) arrays.

        Convention:
          - t=0 at the left edge of the plot area.
          - v=0 at the vertical center of the plot area.
          - Y axis is inverted (lower y_pixel = higher voltage).
        """
        center_y = (grid.plot_y1 + grid.plot_y2) / 2.0
        left_x = float(grid.plot_x1)

        # pixels per division
        px_per_vdiv = grid.cell_h_px  # vertical (voltage)
        px_per_tdiv = grid.cell_w_px  # horizontal (time)

        times = (x_px - left_x) / px_per_tdiv * scale.t_per_div
        voltages = -(y_px - center_y) / px_per_vdiv * scale.v_per_div  # flip sign

        return times, voltages

    # ------------------------------------------------------------------
    # Annotated image for display
    # ------------------------------------------------------------------

    def _annotate(
        self,
        image: np.ndarray,
        grid: GridInfo,
        x_px: np.ndarray,
        y_px: np.ndarray,
    ) -> np.ndarray:
        """Draw the detected trace and grid boundaries onto the image."""
        # Draw plot area bounding box (green)
        cv2.rectangle(
            image,
            (grid.plot_x1, grid.plot_y1),
            (grid.plot_x2, grid.plot_y2),
            color=(0, 200, 80),
            thickness=1,
        )

        # Draw trace overlay (cyan)
        pts = np.column_stack([x_px.astype(int), y_px.astype(int)])
        for i in range(len(pts) - 1):
            cv2.line(image, tuple(pts[i]), tuple(pts[i + 1]), (0, 220, 255), thickness=2)

        return image
