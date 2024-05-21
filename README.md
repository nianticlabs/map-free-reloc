<p align="center">
  <h1 align="center">Map-free Visual Relocalization:<br>Metric Pose Relative to a Single Image</h1>
    <a href="https://earnold.me">Eduardo Arnold</a>
    ·
    <a href="">Jamie Wynn</a>
    ·
    <a href="https://scholar.google.co.uk/citations?user=7wWsNNcAAAAJ">Sara Vicente</a>
    ·
     <a href="https://guiggh.github.io/">Guillermo Garcia-Hernando</a>
    ·
     <a href="https://amonszpart.github.io/">Áron Monszpart</a>
    ·
     <a href="https://www.robots.ox.ac.uk/~victor/">Victor Adrian Prisacariu</a>
    ·
     <a href="https://scholar.google.com/citations?user=ELFm0CgAAAAJ">Daniyar Turmukhambetov</a>
    ·
     <a href="https://twitter.com/eric_brachmann">Eric Brachmann</a>
  </p>
  <h2 align="center">ECCV 2022</h2>
  <h3 align="center"><a href="https://research.nianticlabs.com/mapfree-reloc-benchmark">Project Page</a> | <a href="https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/MapFreeReloc-ECCV22-paper.pdf">Paper</a> | <a href="https://arxiv.org/abs/2210.05494">arXiv</a> | <a href="https://storage.cloud.google.com/niantic-lon-static/research/map-free-reloc/MapFreeReloc-ECCV22-supplemental.pdf">Supplemental</a> </h3> 
  <div align="center"></div>
</p>

This is the reference implementation of the paper **"Map-free Visual Relocalization: Metric Pose Relative to a Single Image"** presented at **ECCV 2022**.

Standard visual relocalization requires hundreds of images and scale calibration to build a scene-specific 3D map. In contrast, we propose Map-free Relocalization, i.e., using only one photo of a scene to enable instant, metric scaled relocalization.

We crowd-sourced a substantial new [dataset](#camera-map-free-visual-relocalization-dataset) for this task, consisting of 655 places. We also define a new benchmark based on this dataset that includes a public [leaderboard](https://research.nianticlabs.com/mapfree-reloc-benchmark).

<p align="center">
    <img src="etc/teaser.png" alt="teaser" width="90%">
</p>

# Overview

1. [Setup](#nut_and_bolt-setup)
1. [Our dataset](#camera-map-free-visual-relocalization-dataset)
1. [Evaluate your method](#bar_chart-evaluate-your-method)
1. [Baselines: Relative Pose Regression](#relative-pose-regression-baselines)
   1. [Single Frame track](#single-frame-track)
   1. [Multi Frame track](#multi-frame-track)
1. [Baselines: Feature Matching + Scale from Estimated Depth](#feature-matching--scale-from-depth-baselines)
1. [Extended Results (7Scenes & Scannet)](#results-on-scannet--7scenes)
1. [Cite](#scroll-cite)
1. [License](#️page_with_curl-license)
1. [Changelog](#pencil-changelog)
1. [Acknowledgements](#octocat-acknowledgements)


# :nut_and_bolt: Setup
Using [Anaconda](https://www.anaconda.com/download/), you can install dependencies with 
```shell
conda env create -f environment.yml
conda activate mapfree
```
We used PyTorch 1.8, PyTorch Lightning 1.6.5, CUDA toolkit 11.1, Python 3.7.12 and Debian GNU/Linux 10.

# :camera: Map-free Visual Relocalization Dataset
We introduce a new [dataset](https://research.nianticlabs.com/mapfree-reloc-benchmark/dataset) for development and evaluation of map-free relocalization. The dataset consists of 655 outdoor scenes, each containing a small ‘place of interest’ such as a sculpture, sign, mural, etc.

To use our code, download [our dataset](https://research.nianticlabs.com/mapfree-reloc-benchmark/dataset) and extract train/val/test.zip files into `data/mapfree`.

## Organization
The dataset is split into 460 training scenes, 65 validation scenes and 130 test scenes.

Each training scene has two sequences of images, corresponding to two different scans of the scene. We provide the absolute pose of each training image, which allows determining the relative pose between any pair of training images.

For validation and test scenes, we provide a single reference image obtained from one scan and a sequence of query images and absolute poses from a different scan.

An exemplar scene contains the following structure:
```
train/
├── s00000
│   ├── intrinsics.txt
│   ├── overlaps.npz
│   ├── poses.txt
│   ├── poses_device.txt
│   ├── seq0
│   │   ├── frame_00000.jpg
│   │   ├── frame_00001.jpg
│   │   ├── frame_00002.jpg
│   │   ├── ...
│   │   └── frame_00579.jpg
│   └── seq1
│       ├── frame_00000.jpg
│       ├── frame_00001.jpg
│       ├── frame_00002.jpg
│       ├── ...
│       └── frame_00579.jpg
```

### **intrinsics.txt**
Encodes per frame intrinsics with format 
```
frame_path fx fy cx cy frame_width frame_height
```

### **poses.txt**
Encodes per frame extrinsics with format 
```
frame_path qw qx qy qz tx ty tz
``` 
where $q$ is the quaternion encoding rotation and $t$ is the **metric** translation vector. 

Note:
- The pose is given in world-to-camera format, i.e. $R(q), t$ transform a world point $p$ to the camera coordinate system as $Rp + t$.
- For val/test scenes, the reference frame (`seq0/frame_00000.jpg`) always has identity pose and the pose of query frames (`seq1/frame_*.jpg`) are given relative to the reference frame. Thus, the absolute pose of a given query frame is equivalent to the relative pose between the reference and the query frames.
- We **DO NOT** provide ground-truth poses for the **test** scenes. These are kept private for evaluation in our [online benchmarking website](https://research.nianticlabs.com/mapfree-reloc-benchmark/). The poses provided for test sequences are invalid lines containing 0 for all parameters.
- There might be "skipped frames", i.e. the linear id of a frame does not necessarily correspond to its frame number. 

### **overlaps.npz**
Available for **training scenes only**, this file provides the overlap score between any (intra- and inter-sequence) pairs of frames and can be used to select training pairs. The overlap score measures the view overlap between two frames as a ratio in the interval $[0,1]$, computed based on the SfM co-visibility. Details of how this is computed is available in the [supplemental materials](https://storage.cloud.google.com/niantic-lon-static/research/map-free-reloc/MapFreeReloc-ECCV22-supplemental.pdf).

The file contains two numpy arrays: 
- `idxs`: stores the sequences and frame numbers for a pair of images (A, B), for which the overlap is computed. Format: `seq_A, frame_A, seq_B, frame_B`
- `overlaps`: which gives the corresponding overlap score. 

For example, to obtain the overlap score between frames `seq0/frame_00023.jpg` and `seq1/frame_00058.jpg` one would do:
```
f = np.load('overlaps.npz', allow_pickle=True)
idxs, overlaps = f['idxs'], f['overlaps']
filter_idx = (idxs == np.array((0, 23, 1, 58))).all(axis=1)
overlap = overlaps[filter_idx]
```

Note:
- Although we computed overlap scores exhaustively between any two pairs, we only provide rows for pairs of frames with non-zero overlap score.
 
### **poses_device.txt**

> **Do not use for the Single Frame leaderboard!**

Contains the device tracking poses from the phone manufacturer's SDK.  
The format is the same as `poses.txt`.  
The validation and test poses have been transformed so the query frame (every 10th starting from index `9`: `9`, `19`, `29`, ...) has identity pose.  
Every 10th frame (index `0`, `10`, `20`, ...) is omitted.

<summary>An example pose file <code>poses_device.txt</code> for scene <code>s00525</code> looks like this:
<details>
<pre><code>seq1/frame_00001.jpg 0.9994922096860480 0.0269317176591893 -0.0157003063652646 0.0065958881785289 0.0466694878078898 -0.0281468845572431 -0.0112877194474297
seq1/frame_00002.jpg 0.9993580756483937 0.0328725115523059 -0.0127593038213454 0.0063273048430992 0.0339232485038949 -0.0285098757477421 -0.0120072983452042
seq1/frame_00003.jpg 0.9994095257784243 0.0337710521318569 -0.0057182100130971 0.0027235813735874 0.0238052857804480 -0.0263538300263913 -0.0098207667922940
seq1/frame_00004.jpg 0.9994142775149452 0.0341439298347318 -0.0013558587396570 0.0018589249041177 0.0177697615576487 -0.0244366729951603 -0.0091893336392181
seq1/frame_00005.jpg 0.9994709356996739 0.0324462544438607 0.0008984342372670 0.0020693187535624 0.0107922389473231 -0.0218151599008894 -0.0084062276379855
seq1/frame_00006.jpg 0.9995967759228006 0.0277079528389628 0.0055091728063658 0.0028642501995992 0.0086483965820601 -0.0177469278559197 -0.0056376986063943
seq1/frame_00007.jpg 0.9997700434443832 0.0209458894930015 0.0026126274935355 0.0037820790767911 0.0037626691655388 -0.0130191227916635 -0.0046983244214344
seq1/frame_00008.jpg 0.9999591047799402 0.0090371065943122 0.0000464505105742 0.0003425119760763 0.0020378531723056 -0.0062499045221460 -0.0016783057659518
seq1/frame_00009.jpg 1.0000000000000000 0.0000000000000000 0.0000000000000000 0.0000000000000000 0.0000000000000000 -0.0000000000000000 0.0000000000000000
seq1/frame_00011.jpg 0.9999122102115333 -0.0028343570013565 0.0028833026511983 0.0126184331870871 -0.0038358715402572 0.0055594114544770 0.0075361352095454
seq1/frame_00012.jpg 0.9998977735666226 -0.0047589344722841 0.0016745278483207 0.0133787486591577 -0.0031548241889184 0.0074063254943168 0.0086534781681345
seq1/frame_00013.jpg 0.9998630580470527 -0.0059850418371746 0.0037297007205538 0.0149710974727477 -0.0027377091384663 0.0057279687266299 0.0071799753997997
seq1/frame_00014.jpg 0.9998568942634775 -0.0067123264981617 0.0044841433737789 0.0148670146626239 -0.0011100149021661 0.0055346086822823 0.0069199077735905
seq1/frame_00015.jpg 0.9999031282964173 -0.0077794581986427 0.0058398554568458 0.0099554076469641 -0.0019798297167677 0.0060388098070261 0.0071602408425884
seq1/frame_00016.jpg 0.9999428133748520 -0.0041002158190476 0.0082137724101253 0.0054856315058257 -0.0017442737456200 0.0055955883320382 0.0062599826478464
seq1/frame_00017.jpg 0.9999532438929408 -0.0004901163942371 0.0095603423341645 0.0013673581676175 -0.0017108482765422 0.0037287971121049 0.0038656792198235
seq1/frame_00018.jpg 0.9999822266759567 -0.0001296402465593 0.0059474062722850 0.0003973464916624 -0.0004313194774018 0.0029779679319886 0.0022206648539896
seq1/frame_00019.jpg 1.0000000000000000 -0.0000000000000000 -0.0000000000000000 -0.0000000000000000 0.0000000000000000 0.0000000000000000 -0.0000000000000000
seq1/frame_00021.jpg 0.9991174014534908 0.0043753050099131 -0.0416160156387965 0.0036581499758310 0.0562188890774214 0.0042611520775912 -0.0473521267483975
seq1/frame_00022.jpg 0.9990560273656551 0.0038494243152731 -0.0429731102874402 0.0050544939430033 0.0558714356884079 0.0034168273536300 -0.0450965029545426
seq1/frame_00023.jpg 0.9990875933607486 0.0021750478867534 -0.0423298259151342 0.0052379191777077 0.0528036499976063 0.0031465186443718 -0.0411179893617076
seq1/frame_00024.jpg 0.9992377263670008 0.0024872418219241 -0.0387361526318281 0.0041581621312520 0.0475880347991984 0.0026831537403738 -0.0362958779137877
seq1/frame_00025.jpg 0.9993967524819487 0.0036629261148744 -0.0344794623038099 0.0019699695560448 0.0398738931715040 0.0024435673020757 -0.0295831119567668
seq1/frame_00026.jpg 0.9995529645105627 -0.0004513363056441 -0.0298478633269434 0.0016650791275553 0.0312575393448512 0.0019572990905338 -0.0227725361872360
seq1/frame_00027.jpg 0.9997698959650141 -0.0040998720240126 -0.0210341558093146 -0.0009541807383061 0.0217680306351583 0.0022364192489630 -0.0162964098631040
seq1/frame_00028.jpg 0.9999206319606756 -0.0048881610389706 -0.0115652795830969 -0.0010392156586288 0.0113113719530353 0.0012417926242774 -0.0088103880189187
seq1/frame_00029.jpg 1.0000000000000000 -0.0000000000000000 0.0000000000000000 0.0000000000000000 0.0000000000000000 0.0000000000000000 0.0000000000000000
seq1/frame_00031.jpg 0.9995818924078186 -0.0134629527639872 0.0009004909140705 -0.0255730011807886 0.1250073985932690 0.0113964897011485 -0.0671276419939781
# ...</code></pre></details></summary>

## Data Loader
We provide a reference PyTorch dataloader for our dataset in [lib/datasets/mapfree.py](lib/datasets/mapfree.py).

# :bar_chart: Evaluate Your Method 
We provide an [online benchmark website](https://research.nianticlabs.com/mapfree-reloc-benchmark/) to evaluate submissions on the test set.  
There are two tracks: [Single Frame](https://research.nianticlabs.com/mapfree-reloc-benchmark/leaderboard?t=single) and [Multi Frame](https://research.nianticlabs.com/mapfree-reloc-benchmark/leaderboard?t=multi9).

Note that, for the **Single Frame** public leaderboard, **we only allow submissions that use single query frames** for their estimates. That is, methods using multi-frame queries are not allowed.  
For the **Multi Frame** public leaderboard, we allow submissions that use up to 9 query frames (very specifically, the query frame and the 8 frames __before__ it) and the provided device tracking poses of those query frames for their estimates.  
Methods using full query scans for their estimates are still **not** allowed.

## Submission Format
The submission file is a ZIP file containing one txt file per scene at its root level without any directories:
```
submission.zip
├── pose_s00525.txt
├── pose_s00526.txt
├── pose_s00527.txt
├── pose_s00528.txt
├── ...
└── pose_s00654.txt
```
Each of the text files should contain the estimated pose for the query frame with the same format as [poses.txt](#posestxt), with the additional `confidence` column: 
```
frame_path qw qx qy qz tx ty tz confidence
```

### Single Frame track

Note that the evaluation for the Single Frame leaderboard only considers every 5th frame of the query sequence, so one does not have to compute the estimated pose for all query frames. This is accounted for in [our dataloader](lib/datasets/mapfree.py#L317).

An example pose file `pose_s00525.txt` for scene `s00525` would look like this:
```
seq1/frame_00000.jpg 0.981085 0.020694 0.191351 0.020694 -1.108672 -0.215504 1.129422 519.7958984375
seq1/frame_00005.jpg 0.976938 0.035391 0.209076 0.025041 -1.198505 -0.254750 1.225280 480.41900634765625
seq1/frame_00010.jpg 0.977999 -0.003629 0.207880 0.017071 -1.139382 -0.119754 1.145658 530.1975708007812
seq1/frame_00015.jpg 0.977930 -0.012163 0.207723 0.018884 -1.132435 -0.119024 0.955460 532.3636474609375
seq1/frame_00020.jpg 0.978110 -0.001814 0.207904 0.008536 -1.157719 -0.121681 0.976792 457.1533813476562
seq1/frame_00025.jpg 0.981478 -0.003330 0.190780 0.017132 -1.154860 -0.121381 1.161221 510.46484375
seq1/frame_00030.jpg 0.963484 -0.004664 0.267198 0.016818 -1.262709 -0.132716 1.269665 518.1480102539062
# ...
```

### Multi Frame track

Note that the evaluation for the Multi Frame leaderboard only considers every 10th frame of the query sequence starting with the 10th (index `9`) frame, so one does not have to compute the estimated pose for all query frames. This is accounted for in [our dataloader](lib/datasets/mapfree.py#L321).

An example pose file `pose_s00525.txt` for scene `s00525` would look like this:
```
seq1/frame_00009.jpg 1 0 0 0 1.0 2 3 100 
seq1/frame_00019.jpg 1 0 0 0 1.1 2 3 200 
seq1/frame_00029.jpg 1 0 0 0 1.2 2 3 300 
seq1/frame_00039.jpg 1 0 0 0 1.3 2 3 400 
# ...
```

## Submission Script
We provide a [submission script](submission.py) to generate submission files:
```shell
python submission.py <config file> [--checkpoint <path_to_model_checkpoint>] -o results/your_method
```

The script reads the configuration of the dataset and the model to determine which track it runs.
To switch between the single and multi-frame setup, configure the `QUERY_FRAME_COUNT` variable in the [Map-free dataset file](config/mapfree.yaml) as:
* QUERY_FRAME_COUNT: 1 # (single frame task) or
* QUERY_FRAME_COUNT: 9 # (multi-frame task)

The model can be also configured accordingly depending on whether it expects single or multiple frames as input. See the [model builder file](lib/builder.py).

The resulting file `results/your_method/submission.zip` can be uploaded to our [online benchmark website](https://research.nianticlabs.com/mapfree-reloc-benchmark/submit) and compared against existing methods in our [leaderboard](https://research.nianticlabs.com/mapfree-reloc-benchmark/leaderboard).

## Local evaluation
We do **NOT** provide ground-truth poses for the test set. But you can still evaluate your method locally, *e.g.* for hyperparameter tuning or model selection, by generating a submission on the **validation set**
```shell
python submission.py <config file> [--checkpoint <path_to_model_checkpoint>] --split val -o results/your_method
```
and evaluate it on the **validation set** using
```shell
python -m benchmark.mapfree results/your_method/submission.zip --split val
```
This is the same script used for evaluation in our benchmarking system, except we use the test set ground-truth poses.

## Examples of submissions for existing baselines
You can generate submissions for the [Relative Pose Regression](#relative-pose-regression-baselines) and [Feature Matching](#feature-matching--scale-from-depth-baselines) baselines using
```shell
# feature matching (SuperPoint+SuperGlue), scale from depth (DPT KITTI), Essential Matrix solver
python submission.py config/matching/mapfree/sg_emat_dptkitti.yaml -o results/sg_emat_dptkitti

# feature matching (LoFTR), scale from depth (DPT NYU), PnP solver
python submission.py config/matching/mapfree/loftr_pnp_dptnyu.yaml -o results/loftr_pnp_dptnyu

# relative pose regression model, 6D rot + 3D trans parametrization
python submission.py config/regression/mapfree/rot6d_trans.yaml --checkpoint weights/mapfree/rot6d_trans.ckpt -o results/rpr_rot6d_trans

# relative pose regression model, 3D-3D correspondence parametrization + Procrustes
python submission.py config/regression/mapfree/3d3d.yaml --checkpoint weights/mapfree/3d3d.ckpt -o results/rpr_3d3d
```
You can explore more methods by inspecting [config/matching/mapfree](config/matching/mapfree) and [config/regression/mapfree](config/regression/mapfree).

# Relative Pose Regression Baselines

##  Pre-trained Models
We provide [Mapfree models](https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/assets/mapfree_rpr_weights.zip) and [Scannet models](https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/assets/scannet_rpr_weights.zip) for all the RPR variants presented in the paper/supplemental.
Extract all weights to `weights/`. The models name match the configuration files in `config/regresion/`

## Custom Models
One can customize the existing models by changing *e.g.* encoder type, feature aggregation variant, output parametrisation and loss functions. All these hyper-parameters are specified in the configuration file for a given model variant. See *e.g.* [config/regression/mapfree/3d3d.yaml](config/regression/mapfree/3d3d.yaml).

We provide multiple variants for the [encoder](lib/models/regression/encoder), [aggregator](lib/models/regression/aggregator.py) and [loss functions](lib/utils/loss.py).

One can also define a custom model by registering it in [lib/models/builder.py](lib/models/builder.py). Given a pair of RGB images, the model must be able to estimate the metric relative pose between the pair of cameras.

## Training a Model
To train a model, use:
```shell
python train.py config/regression/<dataset>/{model variant}.yaml \
                config/{dataset config}.yaml \
                --experiment experiment_name
```
Resume training from a checkpoint by adding `--resume {path_to_checkpoint}`

The top five models, according to validation loss, are saved during training.
Tensorboard results and checkpoints are saved into the folder `weights/experiment_name`.

To switch between the single and multi-frame setup, configure the `QUERY_FRAME_COUNT` variable in the [Map-free dataset file](config/mapfree.yaml) as:
* QUERY_FRAME_COUNT: 1 # (single frame task) or
* QUERY_FRAME_COUNT: 9 # (multi-frame task)

# Feature Matching + Scale from Depth Baselines
We provide different feature matching (SIFT, [SuperPoint+SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork), [LoFTR](https://github.com/zju3dv/LoFTR)), depth regression ([DPT](https://github.com/isl-org/DPT) KITTI, NYU) and pose solver (Essential Matrix Decomposition, PnP) variants.

One can choose the different options for matching, depth and pose solvers by creating a configuration file in [config/matching/mapfree/](config/matching/mapfree/). 

## Download correspondences and depth files
To reproduce feature matching methods baselines
- Download [DPT estimated depth maps](https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/assets/mapfree_dpt_depth.tar.gz).
- Download [feature-matching correspondences](https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/assets/mapfree_correspondences.zip) (LoFTR and SuperPoint+SuperGlue).
- Extract both files to `data/mapfree`

## Custom feature matching method
We provide pre-computed correspondences (SIFT, SuperGlue+SuperPoint and LoFTR) in the path `data/mapfree/{val|test}/{scene}/correspondences_{feature_method}.npz`

To try out your own feature matching methods you need to create a `npz` file storing the correspondences between the reference frame and all query frames for each scene. See steps below:
1. Create a wrapper class to your feature matching method in [etc/feature_matching_baselines/matchers.py](etc/feature_matching_baselines/matchers.py)
2. Add your wrapper into `MATCHERS` in [etc/feature_matching_baselines/compute.py](etc/feature_matching_baselines/compute.py)
3. Execute [etc/feature_matching_baselines/compute.py](etc/feature_matching_baselines/compute.py) using your desired feature matcher on the Mapfree dataset.
4. Create a new configuration file for your feature-matching baseline, *e.g.* modify [config/matching/mapfree/sg_emat_dptkitty.yaml](config/matching/mapfree/sg_emat_dptkitti.yaml) by replacing `SG` in `MATCHES_FILE_PATH` to the name of your matcher.

<details>
<summary> Note on recomputing SG/LoFTR correspondences</summary>

To use SG/LoFTR you need to recursively pull the git submodules using
```shell
git pull --recurse-submodules
```
Then, 
```shell
cd etc/feature_matching_baselines
python compute.py -ds <Scannet or 7Scenes or Mapfree> -m <SIFT or SG or LoFTR>
```
For different 7Scenes pairs variants, include `--pair_txt test_pairs_name.txt`

You also need to download indoor/outdoor weights of LoFTR and extract them to `etc/feature_matching_baselines/weights/`.
</details>

## Custom depth estimation method
We provide estimated **metric depth maps** in `data/mapfree/{val|test}/{scene}/{seq}/frame_{framenum}.dpt{kitti|nyu}.png` (see the [dataset section](#map-free-visual-relocalization))

To try your own depth estimation method you need to provide **metric** depth maps (`png`, encoded in **millimeters**) for each image the the validation/test set.

For example, `data/mapfree/test/s00525/frame_00000.jpg`, will have corresponding depth map `data/mapfree/test/s00525/frame_00000.yourdepthmethod.png`.

To use the custom depth maps, create a new config file, see *e.g.* [config/matching/mapfree/sg_emat_dptkitty.yaml](config/matching/mapfree/sg_emat_dptkitti.yaml), and add the key `ESTIMATED_DEPTH: 'yourdepthmethod'`.

**Externally provided custom depth estimation methods:**
- [KBR depth predictions](https://github.com/jspenmar/slowtv_monodepth#mapfreereloc)

## Custom pose solver
We provide three [pose solvers](lib/models/matching/pose_solver.py): Essential Matrix Decomposition (with metric pose using estimated depth), Perspective-n-Point (PnP) and Procrustes (rigid body transformation given 3D-3D correspondences).

You can add your custom solver to [lib/models/matching/pose_solver.py](lib/models/matching/pose_solver.py) by creating a class that implements `estimate_pose(keypoints0, keypoints1, data)`, where `keypoints` are the image plane coordinates of correspondences and `data` stores all information about the images, including estimated depth maps.

After creating your custom solver class, you need to register it in the [FeatureMatchingModel](lib/models/matching/model.py).

Finally, you can use it by specifying `POSE_SOLVER: 'yourposesolver'` in the configuration file.

# Results on Scannet & 7Scenes
See [this page](benchmark/extended_datasets.md).

# :scroll: Cite
Please cite our work if you find it useful or use any of our code
```latex
@inproceedings{arnold2022mapfree,
      title={Map-free Visual Relocalization: Metric Pose Relative to a Single Image},
      author={Arnold, Eduardo and Wynn, Jamie and Vicente, Sara and Garcia-Hernando, Guillermo and Monszpart, {\'{A}}ron and Prisacariu, Victor Adrian and Turmukhambetov, Daniyar and Brachmann, Eric},
      booktitle={ECCV},
      year={2022},
    }
```

# ️:page_with_curl: License
Copyright © Niantic, Inc. 2022. Patent Pending. All rights reserved. This code is for non-commercial use. Please see the [license file](LICENSE) for terms.

# :pencil: Changelog
- 31/08/2023: updated README.md with externally provdided depthmaps 
- 22/06/2023: updated README.md leaderboard links
- 20/02/2023: benchmark/mapfree.py gives more helpful warnings
- 13/02/2023: updated LICENSE terms

# :octocat: Acknowledgements
We use part of the code from different repositories. We thank the authors and maintainers of the following repositories.
- [CAPS](https://github.com/qianqianwang68/caps)
- [DPT](https://github.com/isl-org/DPT)
- [ExtremeRotation](https://github.com/RuojinCai/ExtremeRotation_code)
- [LoFTR](https://github.com/zju3dv/LoFTR)
- [PlaneRCNN](https://github.com/NVlabs/planercnn)
- [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
- [visloc-relapose](https://github.com/GrumpyZhou/visloc-relapose)
