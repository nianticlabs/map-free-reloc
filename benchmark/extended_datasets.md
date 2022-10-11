# Dataset preparation

## Scannet
- Download the Scannet dataset following the [official instructions](https://github.com/ScanNet/ScanNet#scannet-data).
- Extract the dataset root folder to `data/scannet`
- Download the [Scannet indices](https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/assets/scannet_indices.zip) used for train/val/test splits.
- Download [estimated depth maps and correspondences](https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/assets/scannet_baselines_aux.zip).
- Extract both zip files contents to `data/`
<details>
<summary> Note on Scannet indices</summary>

- The test pairs are the same as SuperGlue/LoFTR (sequences `0707_00 - 0806_00`);
- Training uses SG/LoFTR pairs from sequences `0000_00 - 0699_00`;
- The validation uses the SG/LoFTR pairs from sequences `0700_00 - 0706_00`;
- This split is used to prevent overlapping train/val sequences.
</details>

## 7Scenes
- Download the [7Scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).
- Download [7Scenes pairs indices](https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/assets/sevenscenes_pairs.zip).
- Download [7Scenes feature-matching correspondences](https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/assets/sevenscenes_correspondences.tar.gz).
- Download [7Scenes PlaneRCNN estimated depth maps](https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/assets/sevenscenes_prcnn_depth.zip).
- Extract all zip/tar files to `data/sevenscenes`

## Pre-computed correspondences and depth maps
The pre-computed correspondences (SIFT, SuperGlue+SuperPoint and LoFTR) are found in the path
- Scannet: `data/scannet_misc/correspondences_{feature_method}_scannet_test.npz`
- 7Scenes: `data/sevenscenes/{scene}/correspondences_{feature_method}_test_pairs_{pair_variant}.npz`

The pre-computed depth maps are found in path:
- Scannet (PlaneRCNN monodepth): `data/scannet_misc/scannet_test_depthmaps_planercnn.npz`
- Scannet (DPT NYU monodepth): `data/scannet_misc/scannet_test_depthmaps_dpt.npz`
- 7Scenes (PlaneRCNN monodepth): `data/sevenscenes/{scene}/frame_{framenum}.depth.planercnn.png`

# 📈 Scannet Relative Pose Evaluation
```bash
python -m benchmark.scannet [model config file] [--checkpoint path_to_checkpoint]
```
Each time the script runs, a result file is created in the folder `results` with the same name as the config file. 
This result file contains the rotation and translation errors of each sample in the Scannet test set.
A log text file is also created in `results/log/` with the config file name.

For example, feature-matching methods (more options in [config/matching/scannet](config/matching/scannet)) can be evaluated using:
```bash
#for E-mat based R,t with GT depth maps to get metric pose
python -m benchmark.scannet config/matching/scannet/sift_emat_gt.yaml 

#For E-mat based R,t, with DPT monodepth to get metric pose
python -m benchmark.scannet config/matching/scannet/sift_emat_dpt.yaml

#For PnP based R,t, with PlaneRCNN monodepth to get metric pose
python -m benchmark.scannet config/matching/scannet/sift_pnp_planercnn.yaml

#For Procrustes based R,t, with DPT monodepth to backproject correspondences to 3D
python -m benchmark.scannet config/matching/scannet/sift_procrustes_dpt.yaml
```

# 📈 7Scenes Visual Localisation Evaluation
```bash
python -m benchmark.sevenscenes [model config file] \
                                [dataset config file] \
                                [--checkpoint path_to_checkpoint] \
                                [--test_pair_txt pair_file_name]
```

- Use `config/sevenscenes.yaml` as the dataset config.
- `--test_pair_txt` specifies the pairs of training/query images used in the evaluation (Overrides the one in `config/sevenscenes.yaml`, with default value: `test_pairs.5nn.5cm10m.vlad.minmax.txt` (full EssNetPairs))
- `--one_nn` to filter the single nearest neighbour training image with highest DVLAD similarity to each query image.
- `--triang` uses triangulation (discards translation vector norm) to estimate the absolute pose of the query image
- `--triang_ransac_thres` is the angular inlier threshold for the triangulation RANSAC loop

Note that if neither `--triang` or `--one_nn` is specified, the absolute pose of a query image is computed using all its nearest neighbours.
The absolute pose predictions from each neighbour are aggregated using geometric median of the translation vectors, and the chordal L2 mean of rotation matrices.

Once completed, this evaluation saves the result log as `test_results.txt`.
Additionally, the predicted absolute pose for each query image in a SCENE is saved in a file `pose_7scenes_SCENE.txt`.
Each line in this file follows the format: `image_path qw qx qy qz tx ty tz`, where the quaternion `q` and translation vector `t` encode the predicted absolute pose from world to camera coordinates.

The evaluation code supports feature-matching baselines (SIFT/SuperGlue/LoFTR) for non-metric relative pose (absolute pose obtained via triangulation); and feature-matching & predicted depth, where the metric pose can be obtained using scale from depth.
For example, the baseline SuperGlue + PlaneRCNN depth considering a database of only 10 images per scan, and considering only the closest (DVLAD similarity) database image can be executed with:
```bash
python -m benchmark.sevenscenes \
          config/baseline/sevenscenes/baseline_sg_emat_metric_planercnn_depth.yaml \
          --test_pair_txt test_pairs_ours_km10.txt \
          --one_nn
```
Other baselines, including SIFT/LoFTR are available in `config/matching/sevenscenes/`.
We also provide different test pairs, considering different numbers of database images, namely, `test_pairs_ours_{km1/km2/km5/km10}.txt`.
For each one of these pairs, the database images are selected based on the K-Means clustering of their D-VLAD features.
The pairs file formatting follows the pattern from [EssNet](https://vision.in.tum.de/webshare/u/zhouq/visloc-datasets/README.md).

The other evaluation flags also apply for baselines, for example, one can compute results for SuperGlue + triangulation:
```bash
python -m benchmark.sevenscenes \
          config/baseline/sevenscenes/baseline_sg_emat_metric_planercnn_depth.yaml \
          --test_pair_txt test_pairs_ours_km10.txt \
          --triang
```

Note that the correspondences from feature-matching baselines have been pre-computed for each test pair, and saved in a file for each scene of the 7Scenes dataset.