"""
parameters.py

Scenario 1 path-planning configuration.

New model definition
--------------------
The model file already contains the final planning slowness/cost value.

Flyability rule:
    slowness < 10.0   -> flyable
    slowness >= 10.0  -> no-fly / blocked

Important:
    - Do NOT rebuild risk/emergency/effective cost maps here.
    - DB/DK/FLZ operational point labels can be forced flyable.
    - Do NOT use RA label alone as the blocking rule.
    - The numeric slowness threshold controls flyability for normal grid cells.
"""

from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_FILE = (
    PROJECT_DIR
    / "model"
    / "senario1"
    / "model_senario1_cost_for_pathfinding.xyz"
)


# ============================================================
# Output settings
# ============================================================

DAT_ROOT_DIR = PROJECT_DIR / "output" / "dat" / "senario1"
FIGURE_ROOT_DIR = PROJECT_DIR / "output" / "figures" / "senario1"

PATH_NAME = "path_senario1"


# ============================================================
# Run mode
# ============================================================

# "full"      = load model -> build graph -> run algorithm -> export -> plot
# "plot_only" = skip algorithm and replot from saved path CSV files
RUN_MODE = "full"

# ============================================================
# Possible path connection settings
# ============================================================
# Master switch:
# True  = calculate possible DB/DK/DK-DK path connections and plot figure
# False = skip this step completely in main.py
RUN_POSSIBLE_PATH_CALCULATION = True


SCENARIO_NAME = "senario1"

# Use the same model as the main path-planning workflow.
# Do not point to input/model/... here because MODEL_FILE is already the
# official model defined above.
PATH_INPUT_MODEL = MODEL_FILE

# Save possible connection table and figure inside the existing output tree.
PATH_OUTPUT_CSV = DAT_ROOT_DIR / "paths.csv"
PATH_OUTPUT_FIG = FIGURE_ROOT_DIR / "paths.png"

# Estimate DB -> DK connections.
PATH_INCLUDE_DB_DK = True

# Estimate DK -> DK one-way connections.
PATH_INCLUDE_DK_DK = True

# Optional DB -> DB connections.
# Set True only if you really want DB-DB lines also included.
# For your current request, DB-DK and DK-DK are the main required paths.
PATH_INCLUDE_DB_DB = False

PATH_MAKE_FIGURE = True
PATH_LINE_CMAP = "jet"

# Fancy font
PATH_FANCY_FONT = "DejaVu Serif"
# Other good built-in choices:
# "DejaVu Serif"
# "Times New Roman"   # only if installed

# Smaller colorbar that fits better with figure height
PATH_COLORBAR_SHRINK = 0.82
PATH_COLORBAR_FRACTION = 0.045
PATH_COLORBAR_PAD = 0.02
PATH_COLORBAR_ASPECT = 28
# ============================================================
# Path candidate access check
# ============================================================

# Use the official model columns directly here.
# Do not use PATH_MODEL_COLUMNS = MODEL_COLUMNS if this block appears before MODEL_COLUMNS.
PATH_MODEL_COLUMNS = ("lon", "lat", "z", "slowness", "label")

PATH_SLOWNESS_COLUMN = "slowness"

# Same no-fly rule as path planning.
PATH_NO_FLY_SLOWNESS_THRESHOLD = 10.0
PATH_NO_FLY_THRESHOLD_MODE = "greater_equal"
PATH_NO_FLY_SLOWNESS_TOLERANCE = 1e-9

# DB/DK must have at least one nearby normal flyable grid cell.
# For 50 m grid and 8-neighbor logic, 80 m is good.
PATH_NODE_ACCESS_RADIUS_M = 80.0
PATH_MIN_FLYABLE_NEIGHBORS = 1

# Do not count DB/DK/FLZ labels as normal flyable access cells.
# They can be forced flyable, so counting them may hide a no-fly-covered facility.
PATH_ACCESS_EXCLUDE_PREFIXES = ("DB", "DK", "FLZ")

PATH_NODE_STATUS_CSV = DAT_ROOT_DIR / "path_node_status.csv"

# Background flyable / no-fly dots
PATH_BACKGROUND_MARKER_SIZE = 4.0
PATH_BACKGROUND_ALPHA = 0.55

PATH_STATUS_BOX_LOCATION = "upper right"
# ============================================================
# Algorithm selection
# ============================================================

# Single algorithm:
#     ALGORITHM = "dijkstra"
#
# Multiple algorithms one by one:
#     ALGORITHM = ["dijkstra", "astar"]
#
# Multiple-path algorithm naming rule:
#     src/astar_multiple.py -> output folder:
#     astar/multiple/{MULTIPLE_OUTPUT_VALUE}/
ALGORITHM = ["thetastar"]
# ALGORITHM = ["astar","thetastar"]

# If False, continue to the next algorithm if one fails.
STOP_ON_ALGORITHM_FAILURE = False


# ============================================================
# Start / end nodes
# ============================================================

START_LABEL = "DB01"
END_LABEL = "DK01"

# Optional coordinate-based start/end.
# Use None to use START_LABEL / END_LABEL.
# Expected format depends on your loader, usually:
#     START_COORD = (lon, lat, z)
#     END_COORD   = (lon, lat, z)
START_COORD = None
END_COORD = None


# ============================================================
# Multiple-path comparison control
# ============================================================

# Number of fastest paths to RUN in full mode, or to PLOT in plot_only mode.
A_STAR_K_PATHS = 10

# Folder value to read/write for multiple runs.
#
# Example:
#     A_STAR_K_PATHS = 10
#     MULTIPLE_OUTPUT_VALUE = 100
#
# means:
#     plot/read only fastest 10 paths from old folder multiple/100/
MULTIPLE_OUTPUT_VALUE = 100

# In plot_only mode, limit combined plot to ranks <= this value.
PLOT_MULTIPLE_MAX_RANK = A_STAR_K_PATHS

# "all" means use all ranks after applying PLOT_MULTIPLE_MAX_RANK.
# Or use a list, for example:
#     PLOT_MULTIPLE_RANKS = [1, 2, 5, 10]
PLOT_MULTIPLE_RANKS = "all"

# Plot selected ranked paths into one map with line color = rank.
PLOT_MULTIPLE_RANKED_PATHS = True


# ============================================================
# Multiple-path traveltime histogram
# ============================================================

PLOT_MULTIPLE_TIME_HISTOGRAM = True
PLOT_MULTIPLE_TIME_HISTOGRAM_FASTEST_N = A_STAR_K_PATHS
PLOT_MULTIPLE_TIME_HISTOGRAM_BINS = 20


# ============================================================
# A* / astar_multiple settings
# ============================================================

A_STAR_USE_TURN_PENALTY = True
A_STAR_TURN_WEIGHT = 10.0
A_STAR_TURN_ANGLE_THRESHOLD_DEGREE = 1.0

A_STAR_MAX_EXPANSIONS = 5_000_000
A_STAR_MAX_STATES_PER_NODE_DIRECTION = 150
A_STAR_HEURISTIC_WEIGHT = 1.0

A_STAR_SAVE_ALL_K_PATHS = True
A_STAR_VERBOSE = True

# Parallel settings.
# If A_STAR_N_CORES is None, or larger than machine CPU count,
# use n_cpu - 1 in the algorithm script.
A_STAR_PARALLEL = True
A_STAR_N_CORES = None


# ============================================================
# Model column / value definitions
# ============================================================

# Standard expected model columns:
#     lon lat z slowness label
#
# The fourth column may be called "cost" in discussion, but for travel-time
# path planning it is treated as slowness in s/m.
MODEL_COLUMNS = ("lon", "lat", "z", "slowness", "label")

# Normal flyable slowness value used when you need a reference.
# Your current model uses approximately this for normal flyable nodes.
FLYABLE_SLOWNESS = 0.085

# Hard no-fly threshold.
# New model rule:
#     slowness < 10.0   -> flyable
#     slowness >= 10.0  -> no-fly
NO_FLY_SLOWNESS_THRESHOLD = 10.0
NO_FLY_THRESHOLD_MODE = "greater_equal"

# New explicit blocking switch.
# Your graph-building code should use this to create the flyable mask.
BLOCK_BY_SLOWNESS_THRESHOLD = True

# Tolerance is useful only if your code uses np.isclose somewhere.
# For the new definition, threshold comparison is preferred:
#     slowness >= 10.0
NO_FLY_SLOWNESS_TOLERANCE = 1e-9


# ============================================================
# Graph cost / obstacle rules
# ============================================================

# IMPORTANT:
# Do not block by label in this new model.
# Some normal "N" nodes can be no-fly if their slowness >= 10.
BLOCK_LABEL_PREFIXES = ()

# Do not multiply FLZ again.
# The model already contains the final slowness/cost.
HIGH_COST_LABEL_PREFIXES = ()
FLZ_COST_FACTOR = 1.0

CONNECTIVITY_2D = 8
CONNECTIVITY_3D = 26


# ============================================================
# Force important nodes to be flyable
# ============================================================

# DB/DK/FLZ are operational point facilities.
# They must remain usable even if their point is located on a no-fly
# background/slowness cell. Normal N cells still obey slowness >= 10 => no-fly.
ALWAYS_FLYABLE_PREFIXES = ("DB", "DK", "FLZ")
FORCE_SEARCH_START_END_FLYABLE = True


# ============================================================
# Start/end snapping
# ============================================================

SNAP_START_END_TO_GRID = True

# Snapping may search these labels, but it must still obey the slowness rule.
SNAP_TARGET_PREFIXES = ("N", "FLZ", "DB", "DK")

# New recommended guard:
# snapping must ignore nodes with slowness >= 10.
SNAP_ONLY_TO_FLYABLE = True

INCLUDE_REAL_START_END_IN_OUTPUT = True


# ============================================================
# Endpoint flyable buffer
# ============================================================

# Disable endpoint buffer because it may incorrectly convert blocked cells
# around DB/DK into flyable cells.
ENDPOINT_FLYABLE_BUFFER_RADIUS_M = 0.0
ENDPOINT_FLYABLE_BUFFER_MODE = "both"


# ============================================================
# Slowness cap after model loading
# ============================================================

# Do not cap after loading.
# The model already contains the final slowness/cost field.
CAP_SLOWNESS_AFTER_LOAD = False
SLOWNESS_CAP_VALUE = NO_FLY_SLOWNESS_THRESHOLD


# ============================================================
# Graph neighbor construction
# ============================================================

GRAPH_NEIGHBOR_MODE = "kdtree"

# For regular grid data:
#     1.50 to 1.60 is usually enough to capture immediate neighbors.
KDTREE_RADIUS_FACTOR = 1.60

KDTREE_MAX_NEIGHBORS_2D = 8
KDTREE_MAX_NEIGHBORS_3D = 26


# ============================================================
# Path-step distance output
# ============================================================

SAVE_PATH_STEP_DISTANCE = True
WRITE_EXTRA_PATH_STEP_FILES = True

# If True, path CSV/XYZ contains per-step distance/travel-time columns,
# not only the raw node coordinates.
OVERWRITE_PATH_CSV_XYZ_WITH_STEPS = True


# ============================================================
# Plot settings
# ============================================================

PLOT_MAX_MODEL_POINTS = 300000
PLOT_DPI = 300

PLOT_MODEL_ALPHA = 0.45
PLOT_MODEL_MARKER_SIZE = 1.0
PLOT_PATH_LINE_WIDTH = 1.0

PLOT_REPORT_TEXT_BOX = True

PLOT_INITIATE_FIGURE = True
INITIATE_FIGURE_NAME = f"00_initiate_from_{START_LABEL}_to_{END_LABEL}.png"


# ----------------------------
# Model flyable/no-fly plotting
# ----------------------------

# Plot binary flyable/no-fly using slowness threshold.
PLOT_MODEL_AS_FLYABLE_NOFLY = True

# Do not use label prefixes for no-fly plotting.
PLOT_NO_FLY_PREFIXES = ()

# Plot rule:
#     slowness >= 10.0 -> no-fly
PLOT_NO_FLY_SLOWNESS_THRESHOLD = NO_FLY_SLOWNESS_THRESHOLD
PLOT_NO_FLY_BY_THRESHOLD = True
PLOT_NO_FLY_THRESHOLD_MODE = NO_FLY_THRESHOLD_MODE

# Show DB/DK/FLZ as operational flyable points in plots.
PLOT_ALWAYS_FLYABLE_PREFIXES = ALWAYS_FLYABLE_PREFIXES

# Optional visual overlays.
PLOT_SHOW_FLZ_OVERLAY = True


# ============================================================
# Extra diagnostic plotting
# ============================================================

# Plot a wide figure with:
#   left  = input model as flyable/no-fly
#   right = slowness/cost values with a discrete colorbar
PLOT_INPUT_SLOWNESS_SIDE_BY_SIDE = True
INPUT_SLOWNESS_SIDE_BY_SIDE_NAME = (
    f"00_input_vs_slowness_from_{START_LABEL}_to_{END_LABEL}.png"
)

# Discrete class boundaries for the slowness/cost colorbar.
# The last important boundary is 10.0 because:
#     slowness < 10.0   -> flyable
#     slowness >= 10.0  -> no-fly
#
# The plotting code automatically extends this list if the model contains
# values outside this range.
# None = use 50 discrete equal steps from 0 to NO_FLY_SLOWNESS_THRESHOLD.
# Or provide a custom non-uniform list, e.g. [0, 0.02, 0.05, 0.085, 0.1, 1, 10].
PLOT_SLOWNESS_DISCRETE_BOUNDS = None

PLOT_SLOWNESS_CPT_MODE = "quantile"     # best for real slowness distribution
# PLOT_SLOWNESS_CPT_MODE = "true_values"  # exact values, but can create many CPT intervals
# PLOT_SLOWNESS_CPT_MODE = "uniform"      # simple 0-10 equal steps

PLOT_SLOWNESS_CPT_N_STEPS = 50
PLOT_SLOWNESS_CPT_ROUND_DECIMALS = 4
PLOT_SLOWNESS_CPT_MAX_BOUNDS = 120

# Plot a zoomed corridor around the final selected path.
# This is useful to check whether adjacent cells/nodes around the path
# are flyable or blocked.
PLOT_PATH_ZOOM_DIAGNOSTIC = True

# Corridor half-buffer around the path centerline.
# Use 100-300 m for local node checking; increase if the path is sparse.
PATH_ZOOM_BUFFER_M = 250.0

# Plot graph-neighbor edges connected to path nodes if graph adjacency
# is available from build_grid_graph().
PATH_ZOOM_SHOW_NEIGHBOR_EDGES = True

# Highlight traversable graph nodes directly adjacent to the path.
PATH_ZOOM_SHOW_ADJACENT_NODES = True

# Limit local zoom plotting if the corridor still contains too many points.
PATH_ZOOM_MAX_MODEL_POINTS = 300000


# ============================================================
# Predefined risk map + emergency map + final cost map
# ============================================================

# IMPORTANT:
# The model already includes the final planning slowness/cost.
# Therefore the script should NOT rebuild the risk map, emergency map,
# or effective slowness before graph construction.
USE_PREDEFINED_COSTMAP = False

# Since USE_PREDEFINED_COSTMAP is False, planning uses model["slowness"]
# directly, not model["effective_slowness"].
USE_EFFECTIVE_SLOWNESS_FOR_PLANNING = False

SAVE_COSTMAP_OUTPUTS = False

# Keep True only if your plotting code can plot the already-loaded model
# slowness/cost surface. If it only plots newly computed costmaps,
# set this to False.
PLOT_COSTMAP_OUTPUTS = True

COSTMAP_OUTPUT_NAME = "costmap_senario1"
COSTMAP_FIGURE_SUBDIR = "costmap"
COSTMAP_ROBUST_PERCENTILES = (2, 98)


# ----------------------------
# Risk-map settings
# ----------------------------
# Disabled because the model already contains final slowness/cost.
USE_RISK_MAP = False
BASE_RISK = 0.0

PREFIX_RISK = {}

RISK_COLUMNS = {}

NO_FLY_RISK = 1.0


# ----------------------------
# Emergency-map settings
# ----------------------------
# Disabled because emergency influence is already included in the model.
USE_EMERGENCY_MAP = False

EMERGENCY_PREFIXES = ()

EMERGENCY_DISTANCE_DECAY_M = 1000.0

EMERGENCY_SCORE_COLUMNS = {}

RESTRICTED_PREFIXES_FOR_EMERGENCY = ()


# ----------------------------
# Final cost-map weights
# ----------------------------
# Disabled because final cost/slowness is already in MODEL_FILE.
TRAVEL_WEIGHT = 1.0
RISK_WEIGHT = 0.0
EMERGENCY_WEIGHT = 0.0

MIN_EFFECTIVE_SLOWNESS = 1e-9
MAX_EFFECTIVE_SLOWNESS = None


# ============================================================
# Costmap surface plotting
# ============================================================

# Use pygmt.surface before plotting cost/slowness map.
PLOT_COSTMAP_AS_SURFACE = True

# Surface grid spacing in meters.
# For lon/lat model, the plotting code should convert this to degrees.
COSTMAP_SURFACE_SPACING_M = 20.0


# ============================================================
# Cleanup settings
# ============================================================

RUN_CLEANUP = True
CLEANUP_DRY_RUN = True
CLEANUP_EMPTY_DIRS = True

CLEANUP_TARGET_DIRS = [
    DAT_ROOT_DIR,
    FIGURE_ROOT_DIR,
]

CLEANUP_PATTERNS = [
    "*.tmp",
    "*.temp",
    "*.bak",
    "*.backup",
    "*.log",
    "*.cache",
    "*.gmt",
    "*.cpt",
    "*.grd",
    "*.nc",
    "*.vrt",
    "*.aux.xml",
    "*_tmp.*",
    "*_temp.*",
    "tmp_*",
    "temp_*",
    ".gmt*",
]


# ============================================================
# Helper values for scripts that import parameters.py
# ============================================================

# This expression is the official flyable rule for the new model:
#
#     is_flyable = slowness < FLYABLE_SLOWNESS_MAX
#
# Keep this alias to make downstream code easier to read.
FLYABLE_SLOWNESS_MAX = NO_FLY_SLOWNESS_THRESHOLD

# This expression is the official no-fly rule:
#
#     is_no_fly = slowness >= NO_FLY_SLOWNESS_MIN
#
NO_FLY_SLOWNESS_MIN = NO_FLY_SLOWNESS_THRESHOLD


# ============================================================
# Path zoom direction diagnostic
# ============================================================

# "relative_m" plots x/y as meters from A = lower-left of zoom rectangle.
# This is the best mode when lon/lat differences are too small to inspect.
# "map" keeps original lon/lat or model x/y coordinates.
PATH_ZOOM_COORDINATE_MODE = "relative_m"

# Label path nodes by step number so the movement order is visible.
PATH_ZOOM_LABEL_STEPS = True
PATH_ZOOM_LABEL_STEP_EVERY = 1

# Add arrowheads every N path segments.
PATH_ZOOM_ARROW_EVERY = 1

# Simpler direction check: adjacent nodes are useful, dense neighbor-edge lines are optional.
PATH_ZOOM_SHOW_ADJACENT_NODES = True
PATH_ZOOM_SHOW_NEIGHBOR_EDGES = False
