"""
Interactive globe with Lambert, Mercator and stereographic map projections.

The top half of the window is a rotatable PyVista globe. Right-clicking a
point on the globe makes that location the center of the 2D map below it.
The stereographic projection also has an adjustable view-angle slider.

A Tissot-indicatrix overlay visualizes local angle, length and area distortion:
equal infinitesimal circles on the sphere become circles or ellipses on the map.
This makes the preservation properties of the three projections directly
comparable instead of only describing them in text.
"""


# Standard-library and numerical tools.
import sys
import math
import numpy as np

# PyVista renders the 3D globe; Qt provides the desktop interface.
import pyvista as pv
from pyvista import examples
from pyvistaqt import QtInteractor
from qtpy import QtCore, QtWidgets

# Matplotlib and Cartopy draw the recentered 2D map projections.
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# --- Globe geometry and rendering settings ---
EARTH_RADIUS = 1.0
GRID_RADIUS = 1.003
MARKER_RADIUS = 1.025



# Higher values make the sphere smoother, at the cost of more geometry.
GLOBE_LON_RESOLUTION = 360
GLOBE_LAT_RESOLUTION = 180

# The program starts centered on the intersection of the equator and Greenwich.
INITIAL_LON = 0.0
INITIAL_LAT = 0.0
CAMERA_DISTANCE = 3.15



# Orthographic-camera zoom settings.
GLOBE_VIEW_MARGIN = 1.035
MIN_ZOOM = 0.65
MAX_ZOOM = 4.0
ZOOM_STEP = 1.12




# Slightly enlarge the virtual trackball so dragging near the globe edge feels natural.
ARCBALL_RADIUS_SCALE = 1.08




# Radius used by the map-projection formulas (WGS84 semi-major axis, in metres).
MAP_RADIUS = 6378137.0






# Mercator tends to infinity at the poles, so stop short of +/-90 degrees.
MERCATOR_MAX_ROTATED_LAT = 80.0





# Only stereographic uses an adjustable angular radius.
DEFAULT_VIEW_ANGLE = 90
MIN_VIEW_ANGLE = 30
MAX_VIEW_ANGLE = 150


# --- Distortion visualization settings ---
# Tissot indicatrices start as equal tiny circles on the sphere. Their mapped
# shape shows angular distortion, while their mapped size shows area/scale
# distortion.  The angular radius is deliberately small so the ellipses are a
# good approximation of the local differential distortion.
SHOW_DISTORTION_DEFAULT = True
TISSOT_BASE_RADIUS_DEG = 3.5
TISSOT_CYLINDRICAL_LONGITUDES = (-120, -60, 0, 60, 120)
TISSOT_CYLINDRICAL_LATITUDES = (-60, -30, 0, 30, 60)
TISSOT_STEREO_LONGITUDES = (-120, -60, 0, 60, 120)
TISSOT_STEREO_LATITUDES = (-60, -30, 0, 30, 60)


# Convert latitude to the vertical Mercator coordinate used by this program.
def _mercator_y_degrees(latitude_deg):
    latitude_deg = float(
        np.clip(latitude_deg, -MERCATOR_MAX_ROTATED_LAT, MERCATOR_MAX_ROTATED_LAT)
    )
    latitude = math.radians(latitude_deg)
    return math.degrees(math.log(math.tan(math.pi / 4.0 + latitude / 2.0)))


# Calculate stereographic x/y coordinates and distance from the map centre.
def _stereographic_xy_meters(rotated_lon_deg, rotated_lat_deg):
    
    lon = np.deg2rad(np.asarray(rotated_lon_deg, dtype=float))
    lat = np.deg2rad(np.asarray(rotated_lat_deg, dtype=float))

    cos_lat = np.cos(lat)
    # cos(c) from the spherical law of cosines, because the recentered map centre is at longitude=0, latitude=0.
    cos_c = cos_lat * np.cos(lon)
    denominator = 1.0 + cos_c




    # The denominator reaches zero at the antipode; NumPy is allowed to produce infinities there and the caller clips them out later.
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (2.0 * MAP_RADIUS * cos_lat * np.sin(lon)) / denominator
        y = (2.0 * MAP_RADIUS * np.sin(lat)) / denominator

    angular_distance = np.rad2deg(np.arccos(np.clip(cos_c, -1.0, 1.0)))
    return x, y, angular_distance


def _local_distortion_factors(projection_key, rotated_lon_deg, rotated_lat_deg):
    """Return local linear scale factors and area scale.

    The factors compare an infinitesimal length on the map with the same
    infinitesimal length on the spherical Earth.  ``east`` and ``north`` are
    the principal directions for the cylindrical projections.  Stereographic
    is conformal, so both directions have the same scale factor.
    """
    lon = math.radians(float(rotated_lon_deg))
    lat = math.radians(float(rotated_lat_deg))

    if projection_key == "mercator":
        cos_lat = max(1e-12, abs(math.cos(lat)))
        scale = 1.0 / cos_lat
        return scale, scale, scale * scale

    if projection_key == "stereographic":
        cos_c = math.cos(lat) * math.cos(lon)
        cos_c = float(np.clip(cos_c, -1.0, 1.0))
        c = math.acos(cos_c)
        cos_half = max(1e-12, math.cos(c / 2.0))
        scale = 1.0 / (cos_half * cos_half)
        return scale, scale, scale * scale

    # Lambert cylindrical equal-area: east-west stretching is exactly
    # cancelled by north-south compression, so the area factor is one.
    cos_lat = max(1e-12, abs(math.cos(lat)))
    east_scale = 1.0 / cos_lat
    north_scale = cos_lat
    return east_scale, north_scale, east_scale * north_scale


# -----------------------------------------------------------------------------
# Custom Cartopy projections
# -----------------------------------------------------------------------------

# These classes first rotate the sphere so the selected point becomes the
 # projection centre, then apply the requested map projection.
class RecenteredLambertCylindrical(ccrs.Projection):
    """Lambert cylindrical equal-area projection recentered on any lon/lat."""
    

    # Cylindrical projections can wrap continuously across the date line.
    _wrappable = True

    def __init__(self, central_longitude=0.0, central_latitude=0.0):
        central_longitude = (
            (float(central_longitude) + 180.0) % 360.0
        ) - 180.0
        central_latitude = float(np.clip(central_latitude, -90.0, 90.0))

        # Use a spherical globe so the formulas match the sphere shown above.
        globe = ccrs.Globe(
            ellipse=None,
            semimajor_axis=MAP_RADIUS,
            semiminor_axis=MAP_RADIUS,
        )



        # PROJ's ob_tran performs the recentering rotation before the
        # cylindrical equal-area projection is applied.
        proj4_params = [
            ("proj", "ob_tran"),
            ("o_proj", "cea"),
            ("o_lon_p", 0.0),
            ("o_lat_p", 90.0 - central_latitude),
            ("lon_0", central_longitude),


            ("to_meter", math.radians(1.0) * MAP_RADIUS),
        ]

        super().__init__(proj4_params, globe=globe)

        # Cartopy needs finite bounds even though longitude itself wraps.
        half_height = math.degrees(1.0)
        self.bounds = (-180.0, 180.0, -half_height, half_height)
        self.threshold = 0.5


class RecenteredMercator(ccrs.Projection):
    """Mercator projection recentered on any lon/lat."""


    _wrappable = True

    def __init__(self, central_longitude=0.0, central_latitude=0.0):
        central_longitude = (
            (float(central_longitude) + 180.0) % 360.0
        ) - 180.0
        central_latitude = float(np.clip(central_latitude, -90.0, 90.0))

        globe = ccrs.Globe(
            ellipse=None,
            semimajor_axis=MAP_RADIUS,
            semiminor_axis=MAP_RADIUS,
        )




        # Same recentering idea as Lambert, but followed by Mercator.
        proj4_params = [
            ("proj", "ob_tran"),
            ("o_proj", "merc"),
            ("o_lon_p", 0.0),
            ("o_lat_p", 90.0 - central_latitude),
            ("lon_0", central_longitude),
            ("to_meter", math.radians(1.0) * MAP_RADIUS),
        ]

        super().__init__(proj4_params, globe=globe)

        # Match the visible range to our Mercator latitude cutoff.
        half_height = _mercator_y_degrees(MERCATOR_MAX_ROTATED_LAT)
        self.bounds = (-180.0, 180.0, -half_height, half_height)
        self.threshold = 0.5


class RecenteredStereographic(ccrs.Projection):
    """Stereographic projection centred directly on any lon/lat."""
    

    # Stereographic approaches infinity at the antipode and cannot wrap.
    _wrappable = False

    def __init__(self, central_longitude=0.0, central_latitude=0.0):
        central_longitude = (
            (float(central_longitude) + 180.0) % 360.0
        ) - 180.0
        central_latitude = float(np.clip(central_latitude, -90.0, 90.0))

        globe = ccrs.Globe(
            ellipse=None,
            semimajor_axis=MAP_RADIUS,
            semiminor_axis=MAP_RADIUS,
        )




        # Here Cartopy can use a normal stereographic projection centred
        # directly on the selected longitude and latitude.
        proj4_params = [
            ("proj", "stere"),
            ("lat_0", central_latitude),
            ("lon_0", central_longitude),
            ("k", 1.0),
            ("x_0", 0.0),
            ("y_0", 0.0),
        ]

        super().__init__(proj4_params, globe=globe)

        # Give Cartopy bounds large enough for the maximum slider angle.
        cutoff = math.radians(MAX_VIEW_ANGLE)
        radius = 2.0 * MAP_RADIUS * math.tan(cutoff / 2.0)
        self.bounds = (-radius, radius, -radius, radius)
        self.threshold = 1000.0


# -----------------------------------------------------------------------------
# Main application window
# -----------------------------------------------------------------------------
class InteractiveEarth(QtWidgets.QMainWindow):
    """Qt main window containing the interactive globe and linked 2D map."""
    def __init__(self):
        super().__init__()

        # Current selection and projection-specific state.
        self.setWindowTitle("Interactive Earth + Map Projection Distortion")
        self._marker_actor = None
        self._selected_lon = INITIAL_LON
        self._selected_lat = INITIAL_LAT
        self._view_angle = DEFAULT_VIEW_ANGLE
        self._show_distortion = SHOW_DISTORTION_DEFAULT




        # Camera direction stays fixed; dragging rotates the globe actors instead.
        self._camera_lon = INITIAL_LON
        self._camera_lat = INITIAL_LAT


        # Rotation is stored as a 3x3 matrix and applied to every globe actor.
        self._globe_rotation = np.eye(3, dtype=float)
        self._globe_actors = []
        self._dragging_globe = False
        self._drag_start_vector = None
        self._drag_start_rotation = None
        self._zoom_factor = 1.0

        # Build the interface first, then populate its 3D and 2D views.
        self._build_ui()
        self._draw_globe()
        self._rebuild_map(INITIAL_LON, INITIAL_LAT)
        self._fit_window_to_screen()



        # Run once Qt has finished laying out the widgets so the globe fits correctly.
        QtCore.QTimer.singleShot(0, self._fit_globe_to_viewport)




        # Create the top globe view and lower map view, plus their controls.
    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        instructions = QtWidgets.QLabel(
            "Left-drag to rotate  •  mouse wheel to zoom  •  "
            "right-click a spot to select it"
        )
        instructions.setAlignment(QtCore.Qt.AlignCenter)
        font = instructions.font()
        font.setPointSize(12)
        font.setBold(True)
        instructions.setFont(font)
        root.addWidget(instructions)

        # A vertical splitter lets the user resize the globe and map areas.
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        root.addWidget(self.splitter, 1)


        # ----- 3D globe area -----
        globe_frame = QtWidgets.QFrame()
        globe_layout = QtWidgets.QVBoxLayout(globe_frame)
        globe_layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(globe_frame, lighting="none")
        globe_layout.addWidget(self.plotter.interactor)
        self.splitter.addWidget(globe_frame)



        # Intercept mouse events so we can implement our own rotation and zoom.
        self.plotter.interactor.installEventFilter(self)


        # ----- 2D map area -----
        map_frame = QtWidgets.QFrame()
        map_layout = QtWidgets.QVBoxLayout(map_frame)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(4)

        # Projection selector.
        projection_row = QtWidgets.QHBoxLayout()
        projection_row.setContentsMargins(4, 0, 4, 0)
        projection_row.addStretch(1)
        projection_row.addWidget(QtWidgets.QLabel("Map projection:"))

        self.projection_combo = QtWidgets.QComboBox()
        self.projection_combo.addItem(
            "Lambert cylindrical equal-area", "lambert"
        )
        self.projection_combo.addItem("Mercator conformal ", "mercator")
        self.projection_combo.addItem(
            "Stereographic conformal ", "stereographic"
        )
        self.projection_combo.setToolTip(
            "Lambert preserves area; Mercator and stereographic preserve "
            "angles/local shape."
        )
        projection_row.addWidget(self.projection_combo)
        projection_row.addStretch(1)
        map_layout.addLayout(projection_row)

        # A permanent compact comparison makes the three preservation properties
        # visible at once instead of forcing the user to remember them while
        # switching projections. "Lengths" means preservation of all local lengths.
        self.comparison_label = QtWidgets.QLabel(
            "Lambert: area preserved  •  Mercator: angles preserved  •  "
            "Stereographic: angles preserved  •  all three distort some lengths"
        )
        self.comparison_label.setAlignment(QtCore.Qt.AlignCenter)
        self.comparison_label.setWordWrap(True)
        comparison_font = self.comparison_label.font()
        comparison_font.setPointSize(9)
        self.comparison_label.setFont(comparison_font)
        self.comparison_label.setStyleSheet("color: #52606b;")
        map_layout.addWidget(self.comparison_label)

        # Toggle for the actual distortion visualization. Tissot indicatrices
        # begin as equal small circles on the sphere and show the local image of
        # those circles under the active map projection.
        distortion_row = QtWidgets.QHBoxLayout()
        distortion_row.addStretch(1)
        self.distortion_checkbox = QtWidgets.QCheckBox(
            "Show distortion (Tissot indicatrices)"
        )
        self.distortion_checkbox.setChecked(SHOW_DISTORTION_DEFAULT)
        self.distortion_checkbox.setToolTip(
            "Equal small circles on the sphere. Circles indicate local angle "
            "preservation; ellipses show angular distortion; size change shows "
            "local area/scale distortion."
        )
        distortion_row.addWidget(self.distortion_checkbox)
        distortion_row.addStretch(1)
        map_layout.addLayout(distortion_row)


        # Stereographic-only view-angle controls.
        self.view_angle_widget = QtWidgets.QWidget()
        angle_row = QtWidgets.QHBoxLayout(self.view_angle_widget)
        angle_row.setContentsMargins(14, 0, 14, 0)
        self.view_angle_label = QtWidgets.QLabel("Stereographic view angle:")
        angle_row.addWidget(self.view_angle_label)

        self.view_angle_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.view_angle_slider.setRange(MIN_VIEW_ANGLE, MAX_VIEW_ANGLE)
        self.view_angle_slider.setValue(DEFAULT_VIEW_ANGLE)
        self.view_angle_slider.setSingleStep(1)
        self.view_angle_slider.setPageStep(10)
        self.view_angle_slider.setTickInterval(10)
        self.view_angle_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.view_angle_slider.setToolTip(
            "Angular radius of the stereographic map around the selected center. "
            "90° shows the facing hemisphere; larger values reveal more of the far side."
        )
        angle_row.addWidget(self.view_angle_slider, 1)

        self.view_angle_value = QtWidgets.QLabel(f"{DEFAULT_VIEW_ANGLE}°")
        self.view_angle_value.setMinimumWidth(42)
        self.view_angle_value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        angle_row.addWidget(self.view_angle_value)

        self.view_angle_reset = QtWidgets.QPushButton("Default 90°")
        self.view_angle_reset.setToolTip("Restore the default 90° stereographic view")
        angle_row.addWidget(self.view_angle_reset)
        map_layout.addWidget(self.view_angle_widget)

        # Matplotlib canvas used for all three map projections.
        self.map_figure = Figure(figsize=(8, 4), dpi=100)
        self.map_canvas = FigureCanvas(self.map_figure)

        # The Tissot explanation is drawn inside the Matplotlib figure, in the
        # otherwise unused space to the LEFT of the actual map.  Keeping the
        # canvas at full width means turning Tissot indicatrices on does not move
        # or shrink the map at all.
        map_layout.addWidget(self.map_canvas, 1)
        # Rebuild the map when projection/angle controls change.
        self.projection_combo.currentIndexChanged.connect(
            self._on_projection_changed
        )
        self.view_angle_slider.valueChanged.connect(
            self._on_view_angle_value_changed
        )
        self.view_angle_slider.sliderReleased.connect(
            self._on_view_angle_slider_released
        )
        self.view_angle_reset.clicked.connect(self._reset_view_angle)
        self.distortion_checkbox.toggled.connect(self._on_distortion_toggled)
        self._update_view_angle_controls()




        # Small formula cards are overlaid beside the globe when space allows.
        self._create_globe_math_overlays()
        self._update_globe_math_overlays()

        self.splitter.addWidget(map_frame)

        # Lambert is shorter vertically; the conformal maps get more map space.
        self.splitter.setStretchFactor(0, 14)
        self.splitter.setStretchFactor(1, 10)
        self.splitter.setSizes([560, 400])
        self.splitter.splitterMoved.connect(
            lambda *_: QtCore.QTimer.singleShot(0, self._fit_globe_to_viewport)
        )

        self.setCentralWidget(central)

        # Size the window to most of the available desktop without exceeding it.
    def _fit_window_to_screen(self):
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            self.resize(1000, 850)
            return

        available = screen.availableGeometry()
        width = max(850, int(available.width() * 0.88))
        height = max(700, int(available.height() * 0.88))
        width = min(width, available.width())
        height = min(height, available.height())
        self.resize(width, height)

        x = available.x() + (available.width() - width) // 2
        y = available.y() + (available.height() - height) // 2
        self.move(x, y)




    @staticmethod
        # Qt5 and Qt6 expose mouse coordinates through slightly different APIs.
    def _qt_event_xy(event):
        if hasattr(event, "position"):
            pos = event.position()
            return float(pos.x()), float(pos.y())
        pos = event.pos()
        return float(pos.x()), float(pos.y())

    @staticmethod
        # Compute a camera-up direction that keeps geographic north upright.
    def _camera_up_vector(lon_deg, lat_deg):
        lon = np.deg2rad(lon_deg)
        lat = np.deg2rad(lat_deg)
        up = np.array(
            [
                np.sin(lat) * np.sin(lon),
                -np.sin(lat) * np.cos(lon),
                np.cos(lat),
            ],
            dtype=float,
        )
        norm = np.linalg.norm(up)
        if norm == 0.0:
            return np.array([0.0, 0.0, 1.0])
        return up / norm

        # Point the camera at the origin while keeping north at the top of the screen.
    def _apply_camera_view(self, render=True):
        camera_pos = self._geo_to_xyz(
            self._camera_lon, self._camera_lat, radius=CAMERA_DISTANCE
        )
        view_up = self._camera_up_vector(self._camera_lon, self._camera_lat)

        camera = self.plotter.camera
        camera.position = tuple(camera_pos)
        camera.focal_point = (0.0, 0.0, 0.0)
        camera.up = tuple(view_up)
        camera.roll = 0.0
        self.plotter.reset_camera_clipping_range()

        if render:
            self.plotter.render()

        # Adjust orthographic scale to fit the whole globe at any widget aspect ratio.
    def _fit_globe_to_viewport(self, render=True):
    
        widget = self.plotter.interactor
        width = max(1, int(widget.width()))
        height = max(1, int(widget.height()))
        aspect = width / height





        # On a narrow viewport, increase vertical scale so the globe is not clipped.
        base_scale = EARTH_RADIUS * GLOBE_VIEW_MARGIN
        if aspect < 1.0:
            base_scale /= max(aspect, 1e-6)

        self.plotter.camera.parallel_scale = base_scale / self._zoom_factor

        if hasattr(self, "globe_math_left"):
            self._layout_globe_math_overlays()

        if render:
            self.plotter.render()

        # Create transparent Qt labels that sit on top of the PyVista widget.
    def _create_globe_math_overlays(self):
        
        parent = self.plotter.interactor

        self.globe_math_left = QtWidgets.QLabel(parent)
        self.globe_math_right = QtWidgets.QLabel(parent)

        for label in (self.globe_math_left, self.globe_math_right):
            label.setTextFormat(QtCore.Qt.RichText)
            label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
            label.setStyleSheet(
                "QLabel {"
                " color: rgba(38, 51, 61, 175);"
                " background-color: rgba(255, 255, 255, 178);"
                " border: 1px solid rgba(178, 190, 199, 150);"
                " border-radius: 9px;"
                " padding: 8px 10px;"
                "}"
            )
            label.setWordWrap(False)
            label.raise_()

        self.globe_math_left.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.globe_math_right.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        # Update the side formulas to match the currently selected projection.
    def _update_globe_math_overlays(self):
    
        projection_key = "lambert"
        if hasattr(self, "projection_combo"):
            projection_key = self.projection_combo.currentData() or "lambert"

        if projection_key == "mercator":
            left = (
                "<div style='font-size:11px; letter-spacing:1px;'><b>MERCATOR</b></div>"
                "<div style='font-size:16px; margin-top:4px;'>x = λ′</div>"
                "<div style='font-size:16px;'>"
                "y = ln tan(π/4 + φ′/2)</div>"
            )
            right = (
                "<div style='font-size:11px; letter-spacing:1px;'><b>CONFORMAL</b></div>"
                "<div style='font-size:15px; margin-top:4px;'>k<sub>E</sub> = k<sub>N</sub> = sec φ′</div>"
                "<div style='font-size:10px;'>area factor = sec² φ′</div>"
            )
        elif projection_key == "stereographic":
            left = (
                "<div style='font-size:11px; letter-spacing:1px;'><b>STEREOGRAPHIC</b></div>"
                "<div style='font-size:16px; margin-top:4px;'>"
                "r = 2R tan(c/2)</div>"
                "<div style='font-size:10px;'>c = angular distance from center</div>"
            )
            right = (
                "<div style='font-size:11px; letter-spacing:1px;'><b>CONFORMAL</b></div>"
                "<div style='font-size:15px; margin-top:4px;'>k = sec²(c/2)</div>"
                "<div style='font-size:10px;'>area factor = k²; antipode → ∞</div>"
            )
        else:
            left = (
                "<div style='font-size:11px; letter-spacing:1px;'><b>LAMBERT</b></div>"
                "<div style='font-size:16px; margin-top:4px;'>x = λ′</div>"
                "<div style='font-size:16px;'>y = sin φ′</div>"
            )
            right = (
                "<div style='font-size:11px; letter-spacing:1px;'><b>EQUAL AREA</b></div>"
                "<div style='font-size:15px; margin-top:4px;'>k<sub>E</sub> = sec φ′</div>"
                "<div style='font-size:15px;'>k<sub>N</sub> = cos φ′</div>"
                "<div style='font-size:10px;'>k<sub>E</sub>k<sub>N</sub> = 1</div>"
            )

        self.globe_math_left.setText(left)
        self.globe_math_right.setText(right)
        self.globe_math_left.adjustSize()
        self.globe_math_right.adjustSize()
        self._layout_globe_math_overlays()

        # Place formula cards in unused space beside the globe and hide them if
        # the window is too narrow to fit them cleanly.
    def _layout_globe_math_overlays(self):
        
        if not hasattr(self, "globe_math_left"):
            return

        widget = self.plotter.interactor
        width = max(1, int(widget.width()))
        height = max(1, int(widget.height()))



        try:
            parallel_scale = max(1e-9, float(self.plotter.camera.parallel_scale))
            radius_px = EARTH_RADIUS * height / (2.0 * parallel_scale)
        except Exception:
            radius_px = 0.48 * height

        globe_left = 0.5 * width - radius_px
        globe_right = 0.5 * width + radius_px
        outer_margin = 12
        gap = 14

        self.globe_math_left.adjustSize()
        self.globe_math_right.adjustSize()
        left_w = self.globe_math_left.width()
        left_h = self.globe_math_left.height()
        right_w = self.globe_math_right.width()
        right_h = self.globe_math_right.height()

        enough_left = globe_left - gap >= outer_margin + left_w
        enough_right = width - globe_right - gap >= outer_margin + right_w

        self.globe_math_left.setVisible(enough_left)
        self.globe_math_right.setVisible(enough_right)

        if enough_left:
            left_y = max(outer_margin, int(height * 0.22 - left_h / 2))
            self.globe_math_left.move(outer_margin, left_y)
            self.globe_math_left.raise_()

        if enough_right:
            right_x = width - outer_margin - right_w
            right_y = min(
                height - outer_margin - right_h,
                max(outer_margin, int(height * 0.68 - right_h / 2)),
            )
            self.globe_math_right.move(right_x, right_y)
            self.globe_math_right.raise_()

    @staticmethod
        # Normalize a 3D vector, using a safe fallback for zero-length inputs.
    def _unit_vector(vector, fallback=None):
        
        vector = np.asarray(vector, dtype=float).reshape(3)
        norm = np.linalg.norm(vector)
        if norm > 1e-12:
            return vector / norm
        if fallback is None:
            fallback = (0.0, 0.0, 1.0)
        return np.asarray(fallback, dtype=float).reshape(3)

    @classmethod
        # Rodrigues' formula: build a rotation matrix from an axis and angle.
    def _axis_angle_rotation(cls, axis, angle):
        
        axis = cls._unit_vector(axis)
        x, y, z = axis
        skew = np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float
        )
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)
        return (
            np.eye(3, dtype=float)
            + skew * sin_angle
            + (skew @ skew) * (1.0 - cos_angle)
        )

    @classmethod
        # Find the shortest 3D rotation that takes one unit vector to another.
    def _rotation_between_vectors(cls, source, target):
        
        a = cls._unit_vector(source)
        b = cls._unit_vector(target)
        cross = np.cross(a, b)
        sin_angle = np.linalg.norm(cross)
        cos_angle = float(np.clip(np.dot(a, b), -1.0, 1.0))

        # Parallel vectors need special handling because their cross product is zero.
        if sin_angle < 1e-10:
            if cos_angle > 0.0:
                return np.eye(3, dtype=float)


            # For opposite vectors, choose any stable perpendicular axis for 180 degrees.
            helper = np.array([1.0, 0.0, 0.0])
            if abs(a[0]) > 0.8:
                helper = np.array([0.0, 1.0, 0.0])
            axis = cls._unit_vector(np.cross(a, helper))

            return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=float)

        axis = cross / sin_angle
        x, y, z = axis
        skew = np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float
        )
        return (
            np.eye(3, dtype=float)
            + skew * sin_angle
            + (skew @ skew) * (1.0 - cos_angle)
        )

        # Map a mouse position onto a virtual sphere for arcball-style dragging.
    def _trackball_vector(self, x, y):
        
        widget = self.plotter.interactor
        width = max(1.0, float(widget.width()))
        height = max(1.0, float(widget.height()))


        # Convert the globe's displayed radius from world units to screen pixels.
        parallel_scale = max(1e-9, float(self.plotter.camera.parallel_scale))
        radius_px = EARTH_RADIUS * height / (2.0 * parallel_scale)
        radius_px = max(40.0, radius_px * ARCBALL_RADIUS_SCALE)

        sx = (float(x) - 0.5 * width) / radius_px
        sy = -(float(y) - 0.5 * height) / radius_px
        radius_sq = sx * sx + sy * sy

        # Points inside the trackball lie on the sphere; points outside are
        # clamped to its rim so dragging stays stable.
        if radius_sq <= 1.0:
            sz = np.sqrt(max(0.0, 1.0 - radius_sq))
        else:


            inv = 1.0 / np.sqrt(radius_sq)
            sx *= inv
            sy *= inv
            sz = 0.0

        # Express the trackball vector in world coordinates using the camera basis.
        camera = self.plotter.camera
        camera_pos = np.asarray(camera.position, dtype=float)
        focal = np.asarray(camera.focal_point, dtype=float)
        toward_camera = self._unit_vector(camera_pos - focal)

        up = np.asarray(camera.up, dtype=float)
        up = up - np.dot(up, toward_camera) * toward_camera
        up = self._unit_vector(up, fallback=(0.0, 0.0, 1.0))



        right = self._unit_vector(np.cross(up, toward_camera))
        up = self._unit_vector(np.cross(toward_camera, right))

        world_vector = right * sx + up * sy + toward_camera * sz
        return self._unit_vector(world_vector)

        # Remove unwanted roll after dragging so geographic north remains screen-up.
    def _keep_north_up(self, rotation):
        
        camera = self.plotter.camera
        camera_pos = np.asarray(camera.position, dtype=float)
        focal = np.asarray(camera.focal_point, dtype=float)
        toward_camera = self._unit_vector(camera_pos - focal)

        screen_up = np.asarray(camera.up, dtype=float)
        screen_up = screen_up - np.dot(screen_up, toward_camera) * toward_camera
        screen_up = self._unit_vector(screen_up, fallback=(0.0, 0.0, 1.0))
        screen_right = self._unit_vector(np.cross(screen_up, toward_camera))
        screen_up = self._unit_vector(np.cross(toward_camera, screen_right))





        # Project the rotated north pole onto the screen plane.
        north_world = rotation @ np.array([0.0, 0.0, 1.0], dtype=float)
        north_right = float(np.dot(north_world, screen_right))
        north_up = float(np.dot(north_world, screen_up))

        projection_length = np.hypot(north_right, north_up)
        if projection_length < 1e-8:


            return rotation

        # Rotate around the viewing direction until north has no horizontal component.
        correction_angle = np.arctan2(north_right, north_up)
        correction = self._axis_angle_rotation(toward_camera, correction_angle)
        return correction @ rotation

        # Apply the current rotation matrix to the globe, grid and selected marker.
    def _apply_globe_rotation(self, render=True):
        
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = self._globe_rotation

        for actor in self._globe_actors:
            if actor is not None:
                actor.user_matrix = matrix

        if self._marker_actor is not None:
            self._marker_actor.user_matrix = matrix

        if render:
            self.plotter.render()

        # Finish a drag and clean up numerical drift in the rotation matrix.
    def _end_globe_drag(self):


        # Re-orthogonalize the matrix with SVD so many drags do not slowly deform it.
        u, _, vh = np.linalg.svd(self._globe_rotation)
        self._globe_rotation = u @ vh
        if np.linalg.det(self._globe_rotation) < 0.0:
            u[:, -1] *= -1.0
            self._globe_rotation = u @ vh
        self._apply_globe_rotation(render=False)

        self._dragging_globe = False
        self._drag_start_vector = None
        self._drag_start_rotation = None
        self.plotter.interactor.unsetCursor()

        # Handle resize, drag and wheel events from the PyVista/Qt interactor.
    def eventFilter(self, watched, event):
        
        if watched is self.plotter.interactor:
            event_type = event.type()

            if event_type == QtCore.QEvent.Resize:


                QtCore.QTimer.singleShot(0, self._fit_globe_to_viewport)
                QtCore.QTimer.singleShot(0, self._layout_globe_math_overlays)

            # Left mouse button starts virtual-trackball rotation.
            elif event_type == QtCore.QEvent.MouseButtonPress:
                if event.button() == QtCore.Qt.LeftButton:
                    x, y = self._qt_event_xy(event)
                    self._dragging_globe = True
                    self._drag_start_vector = self._trackball_vector(x, y)
                    self._drag_start_rotation = self._globe_rotation.copy()
                    self.plotter.interactor.setCursor(QtCore.Qt.ClosedHandCursor)
                    return True

            # While dragging, rotate from the start vector to the current vector.
            elif event_type == QtCore.QEvent.MouseMove:
                if (
                    self._dragging_globe
                    and self._drag_start_vector is not None
                    and self._drag_start_rotation is not None
                ):
                    x, y = self._qt_event_xy(event)
                    current_vector = self._trackball_vector(x, y)
                    drag_rotation = self._rotation_between_vectors(
                        self._drag_start_vector, current_vector
                    )
                    candidate_rotation = (
                        drag_rotation @ self._drag_start_rotation
                    )
                    self._globe_rotation = self._keep_north_up(candidate_rotation)

                    self._apply_globe_rotation(render=True)
                    return True

            # Releasing or leaving the widget ends the drag cleanly.
            elif event_type == QtCore.QEvent.MouseButtonRelease:
                if event.button() == QtCore.Qt.LeftButton and self._dragging_globe:
                    self._end_globe_drag()
                    return True

            elif event_type == QtCore.QEvent.Leave:
                if self._dragging_globe:


                    self._end_globe_drag()

            # Mouse wheel changes orthographic zoom rather than camera distance.
            elif event_type == QtCore.QEvent.Wheel:
                delta = event.angleDelta().y()
                if delta:
                    steps = delta / 120.0
                    self._zoom_factor *= ZOOM_STEP ** steps
                    self._zoom_factor = float(
                        np.clip(self._zoom_factor, MIN_ZOOM, MAX_ZOOM)
                    )
                    self._fit_globe_to_viewport(render=True)
                    return True

        return super().eventFilter(watched, event)




    # --- Geographic / Cartesian coordinate conversion ---
    @staticmethod
        # Convert longitude/latitude on the unit sphere to the globe's XYZ convention.
    def _geo_to_xyz(lon_deg, lat_deg, radius=EARTH_RADIUS):
        
        lon = np.deg2rad(lon_deg)
        lat = np.deg2rad(lat_deg)
        cos_lat = np.cos(lat)





        # This axis convention makes lon=0, lat=0 point toward +Y.
        x = -radius * cos_lat * np.sin(lon)
        y = radius * cos_lat * np.cos(lon)
        z = radius * np.sin(lat)
        return np.array([x, y, z], dtype=float)

    @staticmethod
        # Inverse of _geo_to_xyz; radius does not matter, only direction.
    def _xyz_to_geo(point):
        
        point = np.asarray(point, dtype=float).reshape(3)
        norm = np.linalg.norm(point)
        if norm == 0.0:
            return 0.0, 0.0

        x, y, z = point / norm
        lon = np.rad2deg(np.arctan2(-x, y))
        lat = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
        lon = ((lon + 180.0) % 360.0) - 180.0
        return float(lon), float(lat)




    # --- Globe construction and interaction ---
        # Build the textured sphere, lighting, graticule and picking behaviour.
    def _draw_globe(self):
        self.plotter.set_background("#f4f6f8")



        # PyVista sphere geometry used as the textured Earth surface.
        earth = pv.Sphere(
            radius=EARTH_RADIUS,
            theta_resolution=GLOBE_LON_RESOLUTION,
            phi_resolution=GLOBE_LAT_RESOLUTION,
            start_theta=270.001,
            end_theta=270.0,
        )


        # Generate longitude/latitude texture coordinates from each sphere vertex.
        points = earth.points
        uv = np.empty((points.shape[0], 2), dtype=float)
        uv[:, 0] = 0.5 + np.arctan2(-points[:, 0], points[:, 1]) / (2.0 * np.pi)
        uv[:, 1] = 0.5 + np.arcsin(
            np.clip(points[:, 2] / EARTH_RADIUS, -1.0, 1.0)
        ) / np.pi
        earth.active_texture_coordinates = uv



        # PyVista ships with a suitable Earth texture.
        earth_texture = examples.load_globe_texture()

        self.earth_actor = self.plotter.add_mesh(
            earth,
            texture=earth_texture,
            smooth_shading=True,
            ambient=0.28,
            diffuse=0.76,
            specular=0.08,
            specular_power=12,
            pickable=True,
        )
        self._globe_actors.append(self.earth_actor)



        # A headlight follows the camera, keeping the visible hemisphere illuminated.
        headlight = pv.Light(light_type="headlight", intensity=0.92)
        self.plotter.add_light(headlight)



        # Weak fill light prevents the far side of the visible globe looking too harsh.
        fill = pv.Light(
            position=(-0.8, 0.45, 0.55),
            light_type="camera light",
            intensity=0.24,
        )
        self.plotter.add_light(fill)

        # Draw latitude/longitude lines just above the sphere to avoid z-fighting.
        grid = self._build_graticule()
        if grid.n_points:
            self.grid_actor = self.plotter.add_mesh(
                grid,
                color="white",
                line_width=1.0,
                opacity=0.28,
                lighting=False,
                pickable=False,
            )
            self._globe_actors.append(self.grid_actor)

        self._set_marker(INITIAL_LON, INITIAL_LAT)




        # Parallel projection keeps the globe circular instead of adding perspective.
        self.plotter.enable_parallel_projection()
        self._apply_camera_view(render=False)
        self._fit_globe_to_viewport(render=False)
        self._globe_rotation = self._keep_north_up(self._globe_rotation)
        self._apply_globe_rotation(render=False)



        # Right-click surface picking selects a geographic point on the globe.
        self.plotter.enable_surface_point_picking(
            callback=self._on_globe_pick,
            show_message=False,
            show_point=False,
            pickable_window=False,
            left_clicking=False,
            picker="cell",
            clear_on_no_selection=False,
        )
        self.plotter.pickable_actors = [self.earth_actor]
        self.plotter.render()

        # Build latitude and longitude lines as 3D polylines around the sphere.
    def _build_graticule(self):
       
        polylines = []


        # Parallels every 30 degrees between 60 S and 60 N.
        lon_values = np.linspace(-180.0, 180.0, 241)
        for lat in range(-60, 61, 30):
            pts = np.vstack(
                [self._geo_to_xyz(lon, lat, GRID_RADIUS) for lon in lon_values]
            )
            polylines.append(pts)


        # Meridians every 30 degrees around the globe.
        lat_values = np.linspace(-90.0, 90.0, 121)
        for lon in range(-150, 181, 30):
            pts = np.vstack(
                [self._geo_to_xyz(lon, lat, GRID_RADIUS) for lat in lat_values]
            )
            polylines.append(pts)

        return self._polylines_to_mesh(polylines)

    @staticmethod
        # Combine many separate polylines into one efficient PyVista PolyData mesh.
    def _polylines_to_mesh(polylines):
        all_points = []
        line_cells = []
        offset = 0

        # PyVista line cells store: [number_of_points, point_id, point_id, ...].
        for line in polylines:
            line = np.asarray(line, dtype=float)
            if line.ndim != 2 or line.shape[1] != 3 or len(line) < 2:
                continue

            all_points.append(line)
            ids = np.arange(offset, offset + len(line), dtype=np.int64)
            line_cells.append(np.concatenate(([len(line)], ids)))
            offset += len(line)

        if not all_points:
            return pv.PolyData()

        mesh = pv.PolyData(np.vstack(all_points))
        mesh.lines = np.concatenate(line_cells)
        return mesh

        # Move the red selection marker to the requested geographic coordinate.
    def _set_marker(self, lon, lat):
        xyz = self._geo_to_xyz(lon, lat, MARKER_RADIUS)

        if self._marker_actor is not None:
            self.plotter.remove_actor(self._marker_actor, render=False)

        marker = pv.Sphere(
            radius=0.026,
            center=xyz,
            theta_resolution=24,
            phi_resolution=16,
        )
        self._marker_actor = self.plotter.add_mesh(
            marker,
            color="crimson",
            smooth_shading=True,
            pickable=False,
            render=False,
        )
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = self._globe_rotation
        self._marker_actor.user_matrix = matrix

        # Convert a picked, rotated world-space point back to original globe coordinates.
    def _on_globe_pick(self, point):
        
        if point is None:
            return

        # Undo the visual globe rotation before converting XYZ back to lon/lat.
        world_point = np.asarray(point, dtype=float).reshape(3)
        model_point = self._globe_rotation.T @ world_point
        lon, lat = self._xyz_to_geo(model_point)
        self._selected_lon = lon
        self._selected_lat = lat
        self._set_marker(lon, lat)
        self.plotter.render()
        self._rebuild_map(lon, lat)

    # --- Projection controls ---
        # The angle slider only makes sense for the stereographic projection.
    def _update_view_angle_controls(self):
        
        projection_key = "lambert"
        if hasattr(self, "projection_combo"):
            projection_key = self.projection_combo.currentData() or "lambert"
        if hasattr(self, "view_angle_widget"):
            self.view_angle_widget.setVisible(projection_key == "stereographic")

        # Update the label continuously; rebuild only when appropriate.
    def _on_view_angle_value_changed(self, value):
        
        value = int(value)
        self.view_angle_value.setText(f"{value}°")



        if (
        # Keyboard/programmatic changes are applied immediately. During a mouse
        # drag, wait until release so expensive Cartopy redraws do not lag the slider.
            not self.view_angle_slider.isSliderDown()
            and self.projection_combo.currentData() == "stereographic"
        ):
            self._set_view_angle(value)

        # Apply the final slider value after the user releases the handle.
    def _on_view_angle_slider_released(self):
        
        if self.projection_combo.currentData() == "stereographic":
            self._set_view_angle(self.view_angle_slider.value())

        # Clamp and store the stereographic angular radius, then redraw if needed.
    def _set_view_angle(self, value):
        value = int(np.clip(value, MIN_VIEW_ANGLE, MAX_VIEW_ANGLE))
        if value == self._view_angle:
            return
        self._view_angle = value
        if self.projection_combo.currentData() == "stereographic":
            self._rebuild_map(self._selected_lon, self._selected_lat)

        # Restore the standard hemisphere-sized 90 degree stereographic view.
    def _reset_view_angle(self):
        
        if self.view_angle_slider.value() == DEFAULT_VIEW_ANGLE:
            self._view_angle = DEFAULT_VIEW_ANGLE
            if self.projection_combo.currentData() == "stereographic":
                self._rebuild_map(self._selected_lon, self._selected_lat)
        else:
            self.view_angle_slider.setValue(DEFAULT_VIEW_ANGLE)

    def _on_distortion_toggled(self, checked):
        """Show or hide the Tissot distortion overlay and explanation card."""
        self._show_distortion = bool(checked)
        self._rebuild_map(self._selected_lon, self._selected_lat)

        # Change layout/formulas and redraw when the projection selector changes.
    def _on_projection_changed(self, *_):
        
        projection_key = self.projection_combo.currentData() or "lambert"





        # Mercator and stereographic maps are taller, so give them more room.
        if projection_key in ("mercator", "stereographic"):
            self.splitter.setStretchFactor(0, 10)
            self.splitter.setStretchFactor(1, 14)
            self.splitter.setSizes([430, 530])
        else:
            self.splitter.setStretchFactor(0, 14)
            self.splitter.setStretchFactor(1, 10)
            self.splitter.setSizes([560, 400])

        self._update_view_angle_controls()
        self._update_globe_math_overlays()
        self._rebuild_map(self._selected_lon, self._selected_lat)
        QtCore.QTimer.singleShot(0, self._fit_globe_to_viewport)




    # --- 2D map generation ---
        # Rebuild the entire Cartopy map around the currently selected point.
    def _rebuild_map(self, center_lon, center_lat):
        self.map_figure.clear()

        # Read the active projection and stereographic view angle.
        projection_key = "lambert"
        if hasattr(self, "projection_combo"):
            projection_key = self.projection_combo.currentData() or "lambert"
        view_angle = float(getattr(self, "_view_angle", DEFAULT_VIEW_ANGLE))





        # Choose projection, visible bounds, labels and formulas for the side panel.
        if projection_key == "mercator":
            projection = RecenteredMercator(
                central_longitude=center_lon,
                central_latitude=center_lat,
            )
            mercator_lat_extent = MERCATOR_MAX_ROTATED_LAT
            half_height = _mercator_y_degrees(mercator_lat_extent)
            x_limits = (-180.0, 180.0)
            y_limits = (-half_height, half_height)
            map_title = "Mercator conformal map"
            map_subtitle = (
                f"centered at {self._format_lat(center_lat)}, "
                f"{self._format_lon(center_lon)}  •  "
                f"shown to ±{MERCATOR_MAX_ROTATED_LAT:.0f}° rotated latitude"
            )
            math_detail = (
                r"$k_E=k_N=\sec\varphi'$" "\n"
                r"$A_{\rm scale}=\sec^2\varphi'$"
            )
            math_property = "CONFORMAL  •  circles stay circular; size grows toward the poles"
        elif projection_key == "stereographic":
            projection = RecenteredStereographic(
                central_longitude=center_lon,
                central_latitude=center_lat,
            )
            # In stereographic projection, radius on the plane is 2R*tan(c/2).
            cutoff = math.radians(view_angle)
            stereo_radius = 2.0 * MAP_RADIUS * math.tan(cutoff / 2.0)
            x_limits = (-stereo_radius, stereo_radius)
            y_limits = (-stereo_radius, stereo_radius)
            map_title = "Stereographic conformal map"
            map_subtitle = (
                f"centered at {self._format_lat(center_lat)}, "
                f"{self._format_lon(center_lon)}  •  "
                f"view angle {view_angle:.0f}° from center"
            )
            math_detail = (
                r"$k=\sec^2(c/2)$" "\n"
                r"$A_{\rm scale}=k^2$"
            )
            math_property = "CONFORMAL  •  circles stay circular; scale → ∞ at the antipode"
        else:
            projection = RecenteredLambertCylindrical(
                central_longitude=center_lon,
                central_latitude=center_lat,
            )
            half_height = math.degrees(1.0)
            x_limits = (-180.0, 180.0)
            y_limits = (-half_height, half_height)
            map_title = "Lambert cylindrical equal-area map"
            map_subtitle = (
                f"sphere rotated to center {self._format_lat(center_lat)}, "
                f"{self._format_lon(center_lon)}"
            )
            math_detail = (
                r"$k_E=\sec\varphi',\quad k_N=\cos\varphi'$" "\n"
                r"$A_{\rm scale}=k_Ek_N=1$"
            )
            math_property = "EQUAL AREA  •  indicatrices become ellipses but keep equal area"

        # Create a fresh Cartopy axes using the custom recentered projection.
        ax = self.map_figure.add_subplot(1, 1, 1, projection=projection)
        ax.set_global()


        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal", adjustable="box")




        # Stereographic maps are circular, so clip the axes to the visible disc.
        if projection_key == "stereographic":
            unit_circle = MplPath.unit_circle()
            circle_boundary = MplPath(
                unit_circle.vertices * stereo_radius,
                unit_circle.codes,
            )
            ax.set_boundary(circle_boundary, transform=ax.transData)

        # Base map styling.
        ax.set_facecolor("#b8dff2")

        ax.add_feature(
            cfeature.LAND.with_scale("110m"),
            facecolor="#d8d3a7",
            edgecolor="none",
            zorder=1,
        )
        ax.add_feature(
            cfeature.COASTLINE.with_scale("110m"),
            linewidth=0.85,
            zorder=3,
        )
        ax.add_feature(
            cfeature.BORDERS.with_scale("110m"),
            linewidth=0.5,
            alpha=0.72,
            zorder=3,
        )




        # Draw a graticule in the rotated coordinate system. Stereographic needs
        # sampled curves; the cylindrical projections can use straight grid lines.
        if projection_key == "stereographic":
            cutoff = view_angle

            lat_samples = np.linspace(-89.9, 89.9, 721)
            for rotated_lon in range(-150, 181, 30):
                lon_samples = np.full_like(lat_samples, float(rotated_lon))
                x, y, distance = _stereographic_xy_meters(
                    lon_samples, lat_samples
                )
                # Keep only points inside the requested angular radius and away
                # from the antipodal singularity.
                mask = (distance <= cutoff) & np.isfinite(x) & np.isfinite(y)
                if np.count_nonzero(mask) >= 2:
                    ax.plot(
                        x[mask],
                        y[mask],
                        linewidth=0.5,
                        alpha=0.48,
                        linestyle="--",
                        color="black",
                        transform=projection,
                        zorder=2,
                    )

            lon_samples = np.linspace(-179.9, 179.9, 1441)
            for rotated_lat in range(-60, 61, 30):
                lat_samples = np.full_like(lon_samples, float(rotated_lat))
                x, y, distance = _stereographic_xy_meters(
                    lon_samples, lat_samples
                )
                mask = (distance <= cutoff) & np.isfinite(x) & np.isfinite(y)
                if np.count_nonzero(mask) >= 2:
                    ax.plot(
                        x[mask],
                        y[mask],
                        linewidth=0.5,
                        alpha=0.48,
                        linestyle="--",
                        color="black",
                        transform=projection,
                        zorder=2,
                    )
        else:
            for rotated_lon in range(-150, 181, 30):
                ax.plot(
                    [rotated_lon, rotated_lon],
                    [-half_height, half_height],
                    linewidth=0.5,
                    alpha=0.48,
                    linestyle="--",
                    color="black",
                    transform=projection,
                    zorder=2,
                )

            # Horizontal grid lines need the projection-specific y formula.
            if projection_key == "mercator":
                rotated_latitudes = (-80, -60, -30, 0, 30, 60, 80)
                for rotated_lat in rotated_latitudes:
                    if abs(rotated_lat) > mercator_lat_extent + 1e-9:
                        continue
                    y = _mercator_y_degrees(rotated_lat)
                    ax.plot(
                        [-180.0, 180.0],
                        [y, y],
                        linewidth=0.5,
                        alpha=0.48,
                        linestyle="--",
                        color="black",
                        transform=projection,
                        zorder=2,
                    )
            else:
                for rotated_lat in range(-60, 61, 30):
                    y = math.degrees(math.sin(math.radians(rotated_lat)))
                    ax.plot(
                        [-180.0, 180.0],
                        [y, y],
                        linewidth=0.5,
                        alpha=0.48,
                        linestyle="--",
                        color="black",
                        transform=projection,
                        zorder=2,
                    )


        # Tissot indicatrices directly visualize the local differential distortion.
        # They are drawn in the already-rotated projection coordinates, so their
        # axes correspond to infinitesimal east-west/north-south directions.
        if getattr(self, "_show_distortion", SHOW_DISTORTION_DEFAULT):
            self._draw_tissot_indicatrices(
                ax, projection_key, view_angle
            )


        # The selected geographic point is the origin after recentering.
        ax.plot(
            0.0,
            0.0,
            marker="o",
            markersize=7,
            color="crimson",
            markeredgecolor="white",
            markeredgewidth=1.0,
            transform=projection,
            zorder=10,
        )

        # Title shows both the projection type and the current centre.
        ax.set_title(
            map_title + "\n" + map_subtitle,
            fontsize=11,
            pad=8,
        )





        # Leave some room around the axes for the title and formula panel.
        self.map_figure.patch.set_facecolor("#f4f6f8")
        self.map_figure.subplots_adjust(
            left=0.035, right=0.965, bottom=0.055, top=0.82
        )




        # Determine the *actual* map rectangle after Matplotlib has applied the
        # projection aspect ratio.  GeoAxes can still report the full subplot slot
        # before the first draw, which made the old side-panel calculation produce
        # a negative width on some systems.  Drawing once here resolves the final
        # axes geometry without changing the visible layout.
        try:
            self.map_figure.canvas.draw()
        except Exception:
            pass
        ax.apply_aspect()
        map_box = ax.get_position()

        outer_left = 0.018
        outer_right = 0.982
        gap = 0.012
        min_panel_width = 0.060

        # Clamp the measured map box to the figure.  This is defensive against
        # backend-specific rounding and prevents negative add_axes dimensions.
        map_left = float(np.clip(map_box.x0, 0.0, 1.0))
        map_right = float(np.clip(map_box.x1, 0.0, 1.0))
        left_x0 = outer_left
        left_x1 = max(left_x0, map_left - gap)
        right_x0 = min(outer_right, map_right + gap)
        right_x1 = outer_right
        left_space = max(0.0, left_x1 - left_x0)
        right_space = max(0.0, right_x1 - right_x0)

        panel_bottom = max(0.075, float(map_box.y0))
        panel_top = min(0.805, float(map_box.y1))
        panel_height = max(0.18, panel_top - panel_bottom)

        # Put the Tissot explanation into the blank LEFT margin of the figure.
        # It is an independent axes, so the map keeps exactly the same position
        # whether the overlay is enabled or disabled.
        if (
            getattr(self, "_show_distortion", SHOW_DISTORTION_DEFAULT)
            and left_space >= min_panel_width
        ):
            tissot_ax = self.map_figure.add_axes(
                [left_x0, panel_bottom, left_space, panel_height]
            )
            tissot_ax.set_facecolor((1.0, 1.0, 1.0, 0.82))
            tissot_ax.set_xticks([])
            tissot_ax.set_yticks([])
            tissot_ax.set_xlim(0.0, 1.0)
            tissot_ax.set_ylim(0.0, 1.0)
            for spine in tissot_ax.spines.values():
                spine.set_color("#9d8999")
                spine.set_linewidth(0.8)

            tissot_ax.text(
                0.07, 0.92, "TISSOT INDICATRICES",
                ha="left", va="top",
                fontsize=7.4, fontweight="bold", color="#574454",
                transform=tissot_ax.transAxes,
            )
            tissot_ax.text(
                0.07, 0.78,
                "Equal tiny circles on the sphere are\n"
                "projected onto the map.",
                ha="left", va="top",
                fontsize=6.7, color="#574454", linespacing=1.3,
                transform=tissot_ax.transAxes,
            )
            tissot_ax.text(
                0.07, 0.53,
                "Circle  → local angles preserved\n"
                "Ellipse → angular distortion\n"
                "Size change → local scale / area distortion",
                ha="left", va="top",
                fontsize=6.7, color="#574454", linespacing=1.35,
                transform=tissot_ax.transAxes,
            )

        # Projection mathematics stays in the blank RIGHT margin.  On very
        # narrow windows there may genuinely be no side margin; in that case use
        # the larger available margin, but only when it has a valid positive size.
        panel_x0 = right_x0
        panel_width = right_space

        if panel_width < min_panel_width:
            # If Tissot is off, the left margin is free for the math card.
            if (
                not getattr(self, "_show_distortion", SHOW_DISTORTION_DEFAULT)
                and left_space >= min_panel_width
            ):
                panel_x0 = left_x0
                panel_width = left_space
            # If Tissot is on and the right margin is extremely narrow, keep a
            # small but always-valid card at the right edge instead of crashing.
            elif right_space > 0.0:
                panel_x0 = right_x0
                panel_width = right_space
            else:
                panel_width = 0.0

        # add_axes requires strictly non-negative dimensions.  The synchronous
        # draw above should normally leave a real side margin; this tiny fallback
        # merely keeps unusual/narrow backends safe without moving the map.
        panel_width = max(1e-4, float(panel_width))
        panel_height = max(1e-4, float(panel_height))
        panel_x0 = float(np.clip(panel_x0, 0.0, max(0.0, 1.0 - panel_width)))
        panel_bottom = float(
            np.clip(panel_bottom, 0.0, max(0.0, 1.0 - panel_height))
        )

        math_ax = self.map_figure.add_axes(
            [panel_x0, panel_bottom, panel_width, panel_height]
        )
        math_ax.set_facecolor((1.0, 1.0, 1.0, 0.72))
        math_ax.set_xticks([])
        math_ax.set_yticks([])
        math_ax.set_xlim(0.0, 1.0)
        math_ax.set_ylim(0.0, 1.0)
        for spine in math_ax.spines.values():
            spine.set_color("#cbd3da")
            spine.set_linewidth(0.7)

        # Break long formulas into short lines so they remain readable in narrow panels.
        if projection_key == "mercator":




            panel_lines = (
                r"$x=\lambda'_{\rm deg}$",
                r"$y=\frac{180}{\pi}\ln\tan$",
                r"$\left(\frac{\pi}{4}+\frac{\varphi'}{2}\right)$",
            )
        elif projection_key == "stereographic":
            panel_lines = (
                r"$x=\frac{2R\cos\varphi'\sin\lambda'}"
                r"{1+\cos\varphi'\cos\lambda'}$",
                r"$y=\frac{2R\sin\varphi'}"
                r"{1+\cos\varphi'\cos\lambda'}$",
            )
        else:
            panel_lines = (
                r"$x=\lambda'_{\rm deg}$",
                r"$y=\frac{180}{\pi}\sin\varphi'$",
            )

        math_ax.text(
            0.08, 0.91, "PROJECTION MATH",
            ha="left", va="top",
            fontsize=7.1, fontweight="bold", color="#66737d",
            transform=math_ax.transAxes,
        )
        math_ax.text(
            0.08, 0.81, r"$S^2\;\longrightarrow\;\mathbb{R}^2$",
            ha="left", va="top",
            fontsize=11.0, color="#34434e",
            transform=math_ax.transAxes,
        )
        math_ax.text(
            0.08, 0.69,
            "λ′, φ′ = longitude / latitude\nafter recentering",
            ha="left", va="top",
            fontsize=6.8, color="#697680", linespacing=1.25,
            transform=math_ax.transAxes,
        )

        # Space equation lines according to how many lines the selected formula needs.
        equation_y = 0.53
        line_step = 0.115 if len(panel_lines) == 3 else 0.15
        for i, line in enumerate(panel_lines):
            math_ax.text(
                0.08, equation_y - i * line_step, line,
                ha="left", va="top",
                fontsize=8.5, color="#263844",
                transform=math_ax.transAxes,
            )

        math_ax.text(
            0.08, 0.19, math_detail,
            ha="left", va="bottom",
            fontsize=6.9, color="#52606b", linespacing=1.25,
            transform=math_ax.transAxes,
        )
        math_ax.text(
            0.08, 0.055, math_property,
            ha="left", va="bottom",
            fontsize=5.9, fontweight="bold", color="#586671",
            wrap=True, transform=math_ax.transAxes,
        )

        # Queue a repaint without blocking the Qt event loop.
        self.map_canvas.draw_idle()

    def _draw_tissot_indicatrices(
        self, ax, projection_key, view_angle
    ):
        """Draw local distortion ellipses for the active projection.

        Every indicatrix represents the image of the same tiny circle on the
        spherical Earth.  For conformal projections the circles remain circles
        but change size.  For Lambert cylindrical equal-area they become
        ellipses whose area stays constant.
        """
        outline = "#7d2d72"
        fill = (0.55, 0.18, 0.48, 0.12)

        if projection_key in ("lambert", "mercator"):
            base_radius = TISSOT_BASE_RADIUS_DEG
            for rotated_lat in TISSOT_CYLINDRICAL_LATITUDES:
                if (
                    projection_key == "mercator"
                    and abs(rotated_lat) >= MERCATOR_MAX_ROTATED_LAT
                ):
                    continue

                if projection_key == "mercator":
                    y = _mercator_y_degrees(rotated_lat)
                else:
                    y = math.degrees(math.sin(math.radians(rotated_lat)))

                east_scale, north_scale, _ = _local_distortion_factors(
                    projection_key, 0.0, rotated_lat
                )

                for rotated_lon in TISSOT_CYLINDRICAL_LONGITUDES:
                    x = float(rotated_lon)
                    ellipse = Ellipse(
                        (x, y),
                        width=2.0 * base_radius * east_scale,
                        height=2.0 * base_radius * north_scale,
                        facecolor=fill,
                        edgecolor=outline,
                        linewidth=0.9,
                        zorder=6,
                        transform=ax.transData,
                    )
                    ax.add_patch(ellipse)

        else:
            # Stereographic coordinates are in metres, so convert the same
            # angular base radius into an arc length R*dtheta before scaling.
            base_radius = MAP_RADIUS * math.radians(TISSOT_BASE_RADIUS_DEG)
            safe_cutoff = max(0.0, float(view_angle) - 1.5 * TISSOT_BASE_RADIUS_DEG)

            for rotated_lat in TISSOT_STEREO_LATITUDES:
                for rotated_lon in TISSOT_STEREO_LONGITUDES:
                    x, y, distance = _stereographic_xy_meters(
                        rotated_lon, rotated_lat
                    )
                    x = float(np.asarray(x))
                    y = float(np.asarray(y))
                    distance = float(np.asarray(distance))

                    if (
                        not np.isfinite(x)
                        or not np.isfinite(y)
                        or distance > safe_cutoff
                    ):
                        continue

                    scale, _, _ = _local_distortion_factors(
                        projection_key, rotated_lon, rotated_lat
                    )
                    ellipse = Ellipse(
                        (x, y),
                        width=2.0 * base_radius * scale,
                        height=2.0 * base_radius * scale,
                        facecolor=fill,
                        edgecolor=outline,
                        linewidth=0.9,
                        zorder=6,
                        transform=ax.transData,
                    )
                    ax.add_patch(ellipse)


    # --- Small display/cleanup helpers ---
    @staticmethod
        # Format latitude with a hemisphere suffix.
    def _format_lat(lat):
        hemi = "N" if lat >= 0 else "S"
        return f"{abs(lat):.1f}°{hemi}"

    @staticmethod
        # Format longitude with a hemisphere suffix.
    def _format_lon(lon):
        hemi = "E" if lon >= 0 else "W"
        return f"{abs(lon):.1f}°{hemi}"

        # Shut down the embedded PyVista renderer before the Qt window closes.
    def closeEvent(self, event):

        try:
            self.plotter.close()
        finally:
            super().closeEvent(event)


# -----------------------------------------------------------------------------
# Program entry point
# -----------------------------------------------------------------------------
    # Reuse an existing QApplication when embedded; otherwise create our own.
def main():
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    window = InteractiveEarth()
    window.show()

    # Only start the Qt event loop when this function created the application.
    if owns_app:
        exec_method = getattr(app, "exec", None)
        if exec_method is None:
            exec_method = app.exec_
        sys.exit(exec_method())


# Standard Python entry-point guard.
if __name__ == "__main__":
    main()
