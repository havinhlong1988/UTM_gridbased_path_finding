# SPSO node-based route planner

Clean Python project for node-based UAV/UTM route planning using Spherical Vector-based PSO (SPSO)-style particles.

The model is node-based: each node has x/y/z, slowness, and label information. SPSO generates continuous waypoint paths, samples them, snaps samples to the nearest model nodes, and evaluates cost using distance, travel time, no-fly hits, outside-grid hits, smoothness, repeated-node penalty, and multi-route overlap penalty.

## Project layout

```text
main.py
parameters.py
params/SPSO.params
src/SPSO.py
model/senario1/model_senario1_cost_for_pathfinding.xyz
output/SPSO/
```

## Run

```bash
python main.py --params params/SPSO.params
```

Optional command-line overrides:

```bash
python main.py --params params/SPSO.params --n-route 30
python main.py --params params/SPSO.params --n-route 30 --max-overlap 0.15
```

## Input model

Default model path:

```python
MODEL_FILE = "model/senario1/model_senario1_cost_for_pathfinding.xyz"
```

Expected model columns:

```text
x/lon  y/lat  z  slowness  label  [label_prefix]
```

No-fly rule:

```text
slowness >= NOFLY_SLOWNESS_THRESHOLD  -> hard no-fly
slowness <  NOFLY_SLOWNESS_THRESHOLD  -> flyable
```

## Multiple paths and forward/backward lanes

Main controls in `params/SPSO.params`:

```python
N_ROUTE = 30
RUN_FORWARD_PATHS = True
RUN_BACKWARD_PATHS = True
MAX_OVERLAP_RATIO = 0.10
W_OVERLAP = 500000.0
MULTI_PATH_ATTEMPTS_PER_RANK = 3
ENDPOINT_OVERLAP_IGNORE_RADIUS_M = 200.0
OVERLAP_COMPARE_FORWARD_BACKWARD = True
```

For one A-B pair:

```text
N_ROUTE = 30
RUN_FORWARD_PATHS = True
RUN_BACKWARD_PATHS = True
```

means:

```text
A -> B : 30 route alternatives
B -> A : 30 route alternatives
```

## Output files

For each A-B pair, the planner saves:

```text
output/SPSO/routes/A_to_B/A_to_B_multiple_path_summary.csv
output/SPSO/routes/A_to_B/A_to_B_multiple_path_report.png
output/SPSO/routes/A_to_B/A_to_B_multiple_path_report_zoom.png
output/SPSO/routes/A_to_B/A_to_B_forward_path01_nodes.csv
output/SPSO/routes/A_to_B/A_to_B_forward_path01_continuous_points.csv
output/SPSO/routes/A_to_B/A_to_B_forward_path01.png
...
```

The global report is:

```text
output/SPSO/SPSO_route_summary.csv
```

## Zoom-in report figure

The zoom-in figure is controlled by:

```python
PLOT_ZOOM_REPORT = True
ZOOM_MARGIN_M = 250.0
LEGEND_MAX_ROUTES = 12
```

`LEGEND_MAX_ROUTES` only limits the plotted legend so the figure stays readable when `N_ROUTE = 30`. It does not remove routes from the CSV outputs.
