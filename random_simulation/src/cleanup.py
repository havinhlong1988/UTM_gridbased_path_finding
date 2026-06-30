#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cleanup intermediate files generated during the path-finding process.

The cleanup is conservative:
- It removes only files matching temporary/intermediate patterns.
- It protects final output products such as CSV, XYZ, GeoJSON, KML, PNG, JSON.
"""

from __future__ import annotations

from pathlib import Path


PROTECTED_SUFFIXES = {
    ".csv",
    ".xyz",
    ".geojson",
    ".kml",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".svg",
    ".tif",
    ".tiff",
    ".gpkg",
    ".shp",
    ".dbf",
    ".shx",
    ".prj",
}


def cleanup_intermediate_files(
    target_dirs,
    patterns,
    dry_run: bool = False,
    remove_empty_dirs: bool = True,
) -> dict:
    """
    Remove intermediate files from selected folders.

    Parameters
    ----------
    target_dirs : list[str | Path]
        Directories to clean.
    patterns : list[str]
        Glob patterns to remove.
    dry_run : bool
        If True, only report what would be removed.
    remove_empty_dirs : bool
        If True, remove empty folders after file cleanup.
    """

    target_dirs = [Path(d).resolve() for d in target_dirs]
    patterns = list(patterns)

    removed_files = []
    removed_dirs = []
    skipped_files = []
    missing_dirs = []
    errors = []

    for target_dir in target_dirs:
        if not target_dir.exists():
            missing_dirs.append(str(target_dir))
            continue

        if not target_dir.is_dir():
            skipped_files.append(str(target_dir))
            continue

        for pattern in patterns:
            for path in target_dir.rglob(pattern):
                path = path.resolve()

                if not path.exists():
                    continue

                if path.is_dir():
                    continue

                if is_protected_file(path):
                    skipped_files.append(str(path))
                    continue

                try:
                    if dry_run:
                        removed_files.append(str(path))
                    else:
                        path.unlink()
                        removed_files.append(str(path))

                except Exception as exc:
                    errors.append(
                        {
                            "path": str(path),
                            "error": str(exc),
                        }
                    )

    if remove_empty_dirs:
        for target_dir in target_dirs:
            if not target_dir.exists() or not target_dir.is_dir():
                continue

            dirs = sorted(
                [p for p in target_dir.rglob("*") if p.is_dir()],
                key=lambda p: len(p.parts),
                reverse=True,
            )

            for d in dirs:
                try:
                    if any(d.iterdir()):
                        continue

                    if dry_run:
                        removed_dirs.append(str(d))
                    else:
                        d.rmdir()
                        removed_dirs.append(str(d))

                except Exception as exc:
                    errors.append(
                        {
                            "path": str(d),
                            "error": str(exc),
                        }
                    )

    return {
        "dry_run": dry_run,
        "target_dirs": [str(d) for d in target_dirs],
        "patterns": patterns,
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
        "skipped_files": skipped_files,
        "missing_dirs": missing_dirs,
        "errors": errors,
        "n_removed_files": len(removed_files),
        "n_removed_dirs": len(removed_dirs),
        "n_skipped_files": len(skipped_files),
        "n_errors": len(errors),
    }


def is_protected_file(path: Path) -> bool:
    """
    Protect final output files.

    .aux.xml is allowed to be removed.
    """

    name = path.name.lower()

    if name.endswith(".aux.xml"):
        return False

    suffix = path.suffix.lower()

    if suffix in PROTECTED_SUFFIXES:
        return True

    return False


def print_cleanup_summary(summary: dict):
    """
    Print cleanup report.
    """

    print("=" * 70)
    print("CLEANUP SUMMARY")
    print("=" * 70)

    mode = "DRY RUN" if summary.get("dry_run", False) else "DELETE"
    print(f"Mode          : {mode}")
    print(f"Removed files : {summary.get('n_removed_files', 0)}")
    print(f"Removed dirs  : {summary.get('n_removed_dirs', 0)}")
    print(f"Skipped files : {summary.get('n_skipped_files', 0)}")
    print(f"Errors        : {summary.get('n_errors', 0)}")

    if summary.get("missing_dirs"):
        print("\nMissing target dirs:")
        for d in summary["missing_dirs"]:
            print(f"  {d}")

    if summary.get("removed_files"):
        print("\nFiles removed / would be removed:")
        for f in summary["removed_files"][:50]:
            print(f"  {f}")

        if len(summary["removed_files"]) > 50:
            print(f"  ... and {len(summary['removed_files']) - 50} more")

    if summary.get("removed_dirs"):
        print("\nEmpty dirs removed / would be removed:")
        for d in summary["removed_dirs"][:50]:
            print(f"  {d}")

        if len(summary["removed_dirs"]) > 50:
            print(f"  ... and {len(summary['removed_dirs']) - 50} more")

    if summary.get("errors"):
        print("\nCleanup errors:")
        for e in summary["errors"]:
            print(f"  {e['path']}: {e['error']}")