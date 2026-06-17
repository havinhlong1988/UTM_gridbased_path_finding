#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copy final 2D model with labels from make_model project to current project input folder.

Source:
  parent_of_current/make_model/output/02_senario1_no_velocity/04_2D_model_senario_1/
      mixed_model_2d_after_fly_control_for_pathfinding_with_label.xyz

Destination:
  input/model/senario1/
      mixed_model_2d_after_fly_control_for_pathfinding_with_label.xyz
"""

from pathlib import Path
import shutil


# ============================================================
# User settings
# ============================================================

CURRENT_DIR = Path(".").resolve()
PARENT_DIR = CURRENT_DIR.parent

SRC_MODEL = (
    PARENT_DIR
    / "make_model"
    / "output"
    / "02_senario1_no_velocity"
    / "04_2D_model_senario_1"
    / "mixed_model_2d_after_fly_control_for_pathfinding_with_label.xyz"
)

DST_DIR = (
    CURRENT_DIR
    / "model"
    / "senario1"
)

DST_MODEL = DST_DIR / SRC_MODEL.name

# Also create a short standard name if needed.
CREATE_SHORT_COPY = True
SHORT_MODEL_NAME = "model_senario1_with_label.xyz"


# ============================================================
# Main
# ============================================================

def main():
    print("========== COPY INPUT MODEL ==========")
    print(f"Current dir : {CURRENT_DIR}")
    print(f"Parent dir  : {PARENT_DIR}")
    print(f"Source file : {SRC_MODEL}")
    print(f"Output dir  : {DST_DIR}")

    if not SRC_MODEL.exists():
        raise FileNotFoundError(
            "Source model file not found:\n"
            f"  {SRC_MODEL}\n\n"
            "Please check that the source path is correct."
        )

    DST_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(SRC_MODEL, DST_MODEL)

    print(f"[OK] Copied model to: {DST_MODEL}")

    if CREATE_SHORT_COPY:
        short_dst = DST_DIR / SHORT_MODEL_NAME
        shutil.copy2(SRC_MODEL, short_dst)
        print(f"[OK] Copied short-name model to: {short_dst}")

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()