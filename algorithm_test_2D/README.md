# LAE-UTM parameter protocol v2

This version separates algorithm code from algorithm configuration.

## Folder design

```text
project/
├── main.py
├── parameters.py          # common/shared parameters directly here
├── params/
│   ├── astar_multiple.params
│   ├── astar.params
│   ├── FMM2D.params
│   └── RRT.params
└── src/                    # algorithm script
│   ├── astar_multiple.py
│   ├── astar.py
│   ├── FMM2D.py
│   └── RRT.py
├── output/                 # output results
│   ├── dat/            # binary file, map file, xyz report file
│   │   ├── {senario}/            # senario name
│   │       ├── {algorithm}/            # algorithm name
│   │           ├── multiple/   # when run multiple path search    
│   │           |
│   │           |
│   │           ├── path_{senario}_{algorithm}.somehing # fastest path result
│   |    
│   ├── figures/            # figures file, that presenting the files in dat output
│       ├── {senario}/            # senario name
│           ├── {algorithm}/            # algorithm name
│               ├── multiple/   # when run multiple path search    
│               |
│               |
│               ├── path_{senario}_{algorithm}.somehing # fastest path result

```

## Naming rule

The algorithm name in `params/common.params` must match both files:

```text
"FMM2D" -> src/FMM2D.py -> params/FMM2D.params
"astar" -> src/astar.py -> params/astar.params
"RRT"   -> src/RRT.py   -> params/RRT.params
```

## Run control

In `params/common.params`:

```text
PATHFINDING_ALGORITHMS_TO_RUN = ("astar", "FMM2D")
```

This loads:

```text
params/common.params
params/astar.params
params/FMM2D.params
```

## Main.py rule

`main.py` only imports:

```python
import parameters as P
```

Then it calls the dispatcher.

Each algorithm file in `src/` should expose:

```python
def run_from_parameters(P):
    ...
```

So `main.py` does not need to know internal function names like
`run_fmm2d_core`, `astar_multiple`, etc.

## Why .params files?

The `.params` files are plain text config files. They are not executable Python
scripts. This avoids confusion between algorithm code and algorithm parameters.

## Low RAM note

For a 16 GB RAM machine:

```text
LOW_MEMORY_MODE = True
USE_MULTICORE = False
MAX_WORKERS = 1
```

FMM and A* path search normally stores large arrays. Multiprocessing may copy
these arrays and increase RAM use strongly.
