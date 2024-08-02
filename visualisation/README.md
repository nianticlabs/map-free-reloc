### Map-Free Relocalization Visualisation Script

The code in this folder can be used to render a video that shows map-free relocalisation estimates.
If ground truth is available (e.g. for the validation set), both the ground truth and estimated poses will be visualised.
The estimates will be color-coded according to their metric positional error with respect to the ground truth.
In particular, estimates will be green to yellow for up to 1 meter positional error, and red for more than 1 meter error.
If no ground truth is available (e.g. for the test set), only the estimated poses will be visualised.

These videos will look best, if ground truth is available and estimated poses are given for all frames.

The visualisation uses the `pyrender` library, and in particular it's [off-screen rendering capabilities](https://pyrender.readthedocs.io/en/latest/examples/offscreen.html).
The code uses the EGL platform of PyOpenGL. 

We provide an environment file `environment.yml` that can be used to create a conda environment with all necessary dependencies.
To create the environment, run:

```bash
conda env create -f environment.yml
```

Activate the environment via:

```bash
conda activate mapfreevis
```

Call the visualisation script via:

```bash
python render_estimates.py --estimates_path /path/to/estimates --data_path /path/to/data
```

`path/to/estimates` should point to a folder contains the map-free pose files, e.g. `pose_s00460.txt` etc.
`path/to/data` should point to the map-free dataset, e.g. the `test` or `val` folder with scene subfolders `s00460` etc.

The script will iterate through all pose files and create a video for each one. 
All videos will be saved in the folder `renderings`, which can be changed via the `--output_path` argument.

If you want to render a video for a subset of scenes, you can specify them using `--render_subset` followed by a list of scene names, separated by commas, e.g. `--render_subset s00460,s00461`.