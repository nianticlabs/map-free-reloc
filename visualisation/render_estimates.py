#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2024.

import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import logging
import argparse
from pathlib import Path
from render_scene import render_scene

_logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Rendering map-free relocalisation estimates.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--estimates_path', type=Path, required=True,
                        help="Path to the folder that contains file with estimated poses per scene."
                             "That is the folder that contains pose_s00XXX.txt files.")

    parser.add_argument('--data_path', type=Path, required=True,
                        help="Path to the dataset folder, i.e. the s00XXX folders with images.")

    parser.add_argument('--render_subset', type=str,
                        help="Subset of scenes to render, comma separated, e.g. 's00460,s00461'.")

    parser.add_argument('--output_path', type=Path, default=Path('renderings'),
                        help="Path to the folder where the renderings will be saved.")

    parser.add_argument('--confidence_threshold', type=float, default=-1,
                        help="Filter estimates below this confidence threshold.")

    options = parser.parse_args()

    # Get list of all files with estimated poses
    estimates_files = list(options.estimates_path.glob('pose_s*.txt'))

    if len(estimates_files) == 0:
        _logger.error(f"No pose files found in {options.estimates_path}.")
        exit(1)

    # Filter list according to string provided by user
    if options.render_subset:
        # get list of scenes to render
        render_subset = options.render_subset.split(',')
        # only keep files that contain the requested scene ID
        estimates_files = [f for f in estimates_files if f.stem[5:] in render_subset]

    if len(estimates_files) == 0:
        _logger.error(f"No pose files match the requested scene subset: {options.render_subset}.")
        exit(1)

    _logger.info(f"Found {len(estimates_files)} pose files in {options.estimates_path}")

    # do the actual rendering
    for estimates_file in estimates_files:

        # check whether the scene folder exists
        scene_folder = options.data_path / estimates_file.stem[5:]

        if not scene_folder.exists():
            _logger.error(f"Scene folder {scene_folder} does not exist. Skipping.")
            continue

        _logger.info(f"Rendering scene {scene_folder} using estimates from {estimates_file}")
        render_scene(estimates_file, scene_folder, options.output_path, options.confidence_threshold)
