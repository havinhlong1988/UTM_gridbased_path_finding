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

# Store large numbers of per-rank route files here instead of directly in DAT_ROOT_DIR/...
# Example: output/dat/senario1/FMM2D/route_ranks/path_senario1_FMM2D_rank_001.csv
ROUTE_RANKS_SUBDIR = "route_ranks"

# Store one summary plot per undirected facility pair here.
# Example: output/figures/senario1/FMM2D/pair_paths/path_report_FMM2D_pair_DB01_to_DK01.png
PAIR_PATHS_SUBDIR = "pair_paths"
PLOT_FMM2D_PAIR_PATHS = True

# START_LABEL/END_LABEL can be a single string or a list.
# When either one is a list, main.py runs every start/end combination and
# stores each run in output/.../{algorithm}/label_pairs/{START}_to_{END}/.
LABEL_PAIR_RUNS_SUBDIR = "label_pairs"

# In label-pair batch mode, save the zoom-in figure in the same pair figure folder.
# Example:
#   output/figures/senario1/FMM2D/label_pairs/BD1_to_DK4/path_zoom_FMM2D_from_BD1_to_DK4.png
PATH_ZOOM_SAVE_IN_SAME_PAIR_DIR = True
LABEL_PAIR_SKIP_SAME_LABEL = True

# FLZ buffered zone overlay for facility-lane figures.
# Drawn as a transparent blue cover area around every FLZ node.
PLOT_FLZ_BUFFER_M = 200.0
PLOT_FLZ_BUFFER_FACE_COLOR = "#4da3ff"
PLOT_FLZ_BUFFER_ALPHA = 0.22
PLOT_FLZ_BUFFER_EDGE_COLOR = "#1f5fbf"
PLOT_FLZ_BUFFER_EDGE_WIDTH = 0.8


# ============================================================
# Run mode
# ============================================================

# "full"      = load model -> build graph -> run algorithm -> export -> plot
# "plot_only" = skip algorithm and replot from saved path CSV files
RUN_MODE = "full"

# Print elapsed processing time after each algorithm and after the full batch.
PRINT_PROCESSING_TIME = True

# Save a small JSON timing report inside each algorithm output directory.
SAVE_PROCESSING_TIME_JSON = True

# ============================================================
# Possible path connection settings
# ============================================================
# Master switch:
# True  = calculate possible DB/DK/DK-DK path connections and plot figure
# False = skip this step completely in main.py
RUN_POSSIBLE_PATH_CALCULATION = False


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
ALGORITHM = ["astar","dstar","thetastar","fmmastar","dijkstra","floodfill"]
# ALGORITHM = ["astar", "astar_multiple"]

# If False, continue to the next algorithm if one fails.
STOP_ON_ALGORITHM_FAILURE = False


# ============================================================
# Start / end nodes
# ============================================================

START_LABEL = ["BD1","BD2"]
END_LABEL = ["DK1","DK2","DK3","DK4","DK5"]

# Optional endpoint label aliases and shorthand support.
# This lets BD1 resolve to DB01 and DK3 resolve to DK03 if those are the
# real labels in the model. Set to {} if you want exact labels only.
ENDPOINT_LABEL_PREFIX_ALIASES = {"BD": "DB"}
LABEL_PAIR_SKIP_SAME_LABEL = True
LABEL_PAIR_RUNS_SUBDIR = "label_pairs"

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
MULTI_PATH_K_PATHS = 100

# Folder value to read/write for multiple runs.
#
# Example:
#     MULTI_PATH_K_PATHS = 10
#     MULTIPLE_OUTPUT_VALUE = MULTI_PATH_K_PATHS
#
# means:
#     plot/read only fastest 10 paths from old folder multiple/100/
MULTIPLE_OUTPUT_VALUE = MULTI_PATH_K_PATHS

# In plot_only mode, limit combined plot to ranks <= this value.
PLOT_MULTIPLE_MAX_RANK = None

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
PLOT_MULTIPLE_TIME_HISTOGRAM_FASTEST_N = MULTI_PATH_K_PATHS
PLOT_MULTIPLE_TIME_HISTOGRAM_BINS = 20


# ============================================================
# Generic path-search / multiple-path settings
# ============================================================

MULTI_PATH_USE_TURN_PENALTY = True
MULTI_PATH_TURN_WEIGHT = 10.0
MULTI_PATH_TURN_ANGLE_THRESHOLD_DEGREE = 1.0

MULTI_PATH_MAX_EXPANSIONS = 5_000_000
MULTI_PATH_MAX_STATES_PER_NODE_DIRECTION = 150
MULTI_PATH_HEURISTIC_WEIGHT = 1.0

MULTI_PATH_SAVE_ALL_K_PATHS = True
MULTI_PATH_VERBOSE = True


# ============================================================
# Multiple-path overlap control
# ============================================================

# Main switch for path population behavior:
#   "allow"       = old behavior; ranked paths may overlap/share nodes.
#   "non_overlap" = after one path is selected, later paths cannot reuse
#                   previous path nodes/edges except inside the allowed
#                   buffer around start/end and DB/DK/FLZ service zones.
MULTI_PATH_OVERLAP_MODE = "non_overlap"

# Overlap is still allowed within this radius around:
#   - search start/root
#   - search destination
#   - DB/DK/FLZ nodes
# Use 100-200 m for a 50 m grid. Increase if DB/DK/FLZ service areas need
# a wider common access corridor.
MULTI_PATH_NON_OVERLAP_BUFFER_RADIUS_M = 150.0

# Facility labels where overlap is operationally acceptable.
MULTI_PATH_NON_OVERLAP_ALLOWED_PREFIXES = ("DB", "DK", "FLZ")

# Also block reused edges, not only reused nodes. Keep True for strict separation.
MULTI_PATH_NON_OVERLAP_BLOCK_EDGES = True

# Parallel settings.
# If MULTI_PATH_N_CORES is None, or larger than machine CPU count,
# use n_cpu - 1 in the algorithm script.
MULTI_PATH_PARALLEL = True
MULTI_PATH_N_CORES = 10

# Parallel mode for non-overlap path population:
#   "sequential"      = one exact search, lock path, repeat.
#   "candidate_pool"  = for each rank, run several candidate searches in
#                       parallel, accept the best valid non-overlap candidate,
#                       then lock its nodes/edges before the next rank.
MULTI_PATH_PARALLEL_MODE = "candidate_pool"

# Number of candidate searches launched for each path rank.
# Usually set close to the number of CPU cores you want to use.
MULTI_PATH_CANDIDATES_PER_ROUND = 10

# If one candidate round cannot find a usable non-overlap path, retry with
# stronger diversity up to this many rounds for the same rank.
MULTI_PATH_MAX_ROUNDS_PER_PATH = 3

# Search-only diversity penalty used by candidate_pool workers.
# 0.0 = all workers search the same exact best path.
# 0.10-0.50 usually gives useful alternative corridors.
# Returned path costs are still reported using the true original cost.
MULTI_PATH_CANDIDATE_DIVERSITY_WEIGHT = 0.5

# Fixed seed keeps results reproducible between runs.
MULTI_PATH_CANDIDATE_SEED = 20260618

# Optional per-worker expansion limit. None means use MULTI_PATH_MAX_EXPANSIONS.
MULTI_PATH_MAX_EXPANSIONS_PER_CANDIDATE = None


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

INCLUDE_REAL_START_END_IN_OUTPUT = False


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

PLOT_INITIATE_FIGURE = False
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
PLOT_INPUT_SLOWNESS_SIDE_BY_SIDE = False
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


# ============================================================
# FMM2D all-facility fastest-path plotting mode
# ============================================================
# START_LABEL=None and END_LABEL=None means FMM2D should compute/return
# all requested facility pairs instead of one selected start/end pair.
FMM2D_PAIR_MODE = "facility_library"
FMM2D_PAIR_RETURN_MODE = "all"

# Build base unordered routes. In path_offset mode, reverse-direction
# main/backup paths are generated explicitly for every base route.
FMM2D_PAIR_INCLUDE_DB_DK = True
FMM2D_PAIR_INCLUDE_DB_DB = True
FMM2D_PAIR_INCLUDE_DK_DK = True
FMM2D_PAIR_INCLUDE_REVERSE = False
FMM2D_PAIR_DEDUP_TWO_WAY = True

# Remove invalid/self paths.
FMM2D_PAIR_SKIP_SAME_LABEL = True
FMM2D_PAIR_SKIP_SAME_COORD = True
FMM2D_PAIR_SAME_COORD_TOLERANCE_M = 1.0
FMM2D_PAIR_MIN_DISTANCE_M = 50.0

# Save every path returned by FMM2D into combined report files.
MULTI_PATH_SAVE_ALL_K_PATHS = True
PLOT_MULTIPLE_RANKED_PATHS = True
PLOT_MULTIPLE_RANKS = "all"
PLOT_MULTIPLE_MAX_RANK = None




# ============================================================
# FMM2D spatial collision avoidance / path-offset routing
# ============================================================
# Time-offset collision avoidance has been removed.
# New behavior:
#   For every base route A--B, generate:
#     A -> B main
#     A -> B backup
#     B -> A main
#     B -> A backup
# Paths are spatially separated by a path-offset buffer. Overlap is allowed
# only inside the service-zone buffer around DB/DK/FLZ and route endpoints.
# If strict separation cannot produce all 4 paths, the fallback creates the
# smallest route-level traffic link and marks the affected path(s).
FMM2D_COLLISION_AVOIDANCE_MODE = "path_offset"

# Number of requested paths per direction.
FMM2D_PATH_OFFSET_FORWARD_PATHS = 2
FMM2D_PATH_OFFSET_BACKWARD_PATHS = 2

# Hard spatial separation distance between route alternatives.
FMM2D_PATH_OFFSET_BUFFER_M = 200.0

# Main/backup lane-pair preference.
# The backup lane is still not allowed to overlap the main lane outside the
# path-offset buffer, but it receives a soft penalty when it goes too far away
# from the same-direction main lane. This makes the backup usable for lane
# switching during collision avoidance. Forward and backward directions each
# get their own independent main/backup pair; they do not need to stay close.
FMM2D_LANE_PAIR_CLOSE_PARALLEL_PRIORITY = True
FMM2D_LANE_PAIR_PREFERRED_MAX_DISTANCE_M = 450.0
FMM2D_LANE_PAIR_DISTANCE_WEIGHT = 1.5
FMM2D_LANE_PAIR_MAX_PENALTY_FACTOR = 25.0

# Optional hard maximum distance from backup lane to the same-direction main lane.
# None keeps it as a soft priority only, which is safer when no-fly cells are tight.
FMM2D_LANE_PAIR_HARD_MAX_DISTANCE_M = None
FMM2D_LANE_PAIR_HARD_LIMIT_FOR_TRAFFIC_LINK = False

# Overlap exception zone around DB/DK/FLZ and route endpoints.
FMM2D_PATH_OFFSET_ALLOWED_BUFFER_M = 200.0
FMM2D_PATH_OFFSET_ALLOWED_PREFIXES = ("DB", "DK", "FLZ")

# Try strict non-overlap first. If it fails, allow a penalized shared corridor
# and tag that corridor as a traffic link.
FMM2D_PATH_OFFSET_STRICT_BEFORE_TRAFFIC_LINK = True
FMM2D_TRAFFIC_LINK_BUFFER_M = 200.0
FMM2D_TRAFFIC_LINK_PENALTY_FACTOR = 50.0
FMM2D_TRAFFIC_LINK_MINIMIZE = True

# The old schedule/time-offset figure is disabled because path_offset is a
# spatial routing solution, not a departure-delay solution.
FMM2D_PLOT_COLLISION_TIME_REPORT = False
FMM2D_COLLISION_TIME_REPORT_MAX_PATHS = None


# ============================================================
# Path-offset facility-plot style
# ============================================================
# Direction background underlay.
# Forward = A -> B side, Backward = B -> A side.
PLOT_PATH_OFFSET_FORWARD_BG_COLOR = "yellow"     # yellow
PLOT_PATH_OFFSET_BACKWARD_BG_COLOR = "#d9d9d9"  # light gray
PLOT_PATH_OFFSET_DIRECTION_BG_ALPHA = 0.32
PLOT_PATH_OFFSET_DIRECTION_BG_WIDTH_FACTOR = 5.5
PLOT_PATH_OFFSET_DIRECTION_BG_MIN_WIDTH = 5.5

# Main/backup line style.
# Main uses solid lines; backup uses dashed lines.
PLOT_PATH_OFFSET_BACKUP_DASH_PATTERN = (6, 4)

# Extra fallback shared-corridor buffer for traffic-link paths.
PLOT_PATH_OFFSET_TRAFFIC_LINK_BG_ALPHA = 0.18
PLOT_PATH_OFFSET_TRAFFIC_LINK_WIDTH_FACTOR = 7.0
PLOT_PATH_OFFSET_TRAFFIC_LINK_MIN_WIDTH = 6.0


# ============================================================
# Algorithm-specific parameter loader
# ============================================================
# Common/shared settings stay in this file.  Only algorithm-specific settings
# are stored in params/{ALGORITHM}.params, for example:
#     src/FMM2D.py  <->  params/FMM2D.params
#     src/astar.py  <->  params/astar.params
#
# The loader preserves case, but it also performs a case-insensitive fallback
# so ALGORITHM = ["fmm2d"] can still find params/FMM2D.params if that is the
# only matching file.

PARAMS_DIR = PROJECT_DIR / "params"
LOADED_ALGORITHM_PARAM_FILES = []


def _as_algorithm_list(value):
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _resolve_params_file(algorithm_name):
    name = str(algorithm_name).strip()
    if not name:
        return None

    exact = PARAMS_DIR / f"{name}.params"
    if exact.exists():
        return exact

    try:
        for candidate in PARAMS_DIR.glob("*.params"):
            if candidate.stem.lower() == name.lower():
                return candidate
    except Exception:
        pass

    return None


def _load_algorithm_params():
    global LOADED_ALGORITHM_PARAM_FILES
    LOADED_ALGORITHM_PARAM_FILES = []

    for algorithm_name in _as_algorithm_list(ALGORITHM):
        param_file = _resolve_params_file(algorithm_name)
        if param_file is None:
            # Not every algorithm must have a params file.
            continue

        code = param_file.read_text(encoding="utf-8")
        exec(compile(code, str(param_file), "exec"), globals(), globals())
        LOADED_ALGORITHM_PARAM_FILES.append(str(param_file))



# ============================================================
# Global resource / low-RAM settings
# ============================================================
# Shared defaults used by algorithm-specific params files.
# Keep conservative for a 16 GB RAM machine.
LOW_MEMORY_MODE = True
USE_MULTICORE = False
MAX_WORKERS = 1


_load_algorithm_params()
