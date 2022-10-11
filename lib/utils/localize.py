# Heavily based on https://github.com/GrumpyZhou/visloc-relapose/blob/master/utils/eval/localize.py
import os
import itertools
import time
import warnings

import numpy as np
from scipy.spatial.distance import cdist, euclidean
from scipy.spatial.transform import Rotation
from transforms3d.quaternions import quat2mat, mat2quat
import matplotlib.pyplot as plt


def cal_vec_angle_error(label, pred, eps=1e-10):
    assert len(label.shape) == len(pred.shape)

    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)

    v1 = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    v2 = label / np.linalg.norm(label, axis=1, keepdims=True)
    d = np.around(np.sum(np.multiply(v1, v2), axis=1, keepdims=True),
                  decimals=4)  # around to 0.0001 can assure d <= 1
    d = np.clip(d, a_min=-1, a_max=1)
    error = np.degrees(np.arccos(d))

    # If vector are all zero leads to zero division
    # currently set such cases to error = 0.
    error[np.where(np.isnan(error))] = 0.0
    return error


def cal_quat_angle_error(label, pred):
    assert label.shape == (4,)
    assert pred.shape == (4,)

    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)
    q1 = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    q2 = label / np.linalg.norm(label, axis=1, keepdims=True)
    d = np.abs(np.sum(np.multiply(q1, q2), axis=1, keepdims=True))
    d = np.clip(d, a_min=-1, a_max=1)
    error = 2 * np.degrees(np.arccos(d))
    return error


def save_results_visualisation(file_path):
    results_dict = np.load(file_path, allow_pickle=True).item()
    out_file_path = os.path.join(os.path.split(file_path)[0], 'pose_')

    for scene, scene_res in results_dict.items():
        with open(out_file_path+scene+'.txt', 'w') as f:
            for test_im, res in scene_res.items():
                # skip image, if failure
                if res is None:
                    continue
                abs_pose = res['abs_pose_pred']
                inliers = res['inliers']
                formatter = {'float': lambda v: f'{v:.6f}'}
                max_line_width = 1000
                q_str = np.array2string(abs_pose.q, formatter=formatter,
                                        max_line_width=max_line_width)[1:-1]
                t_str = np.array2string(abs_pose.t, formatter=formatter,
                                        max_line_width=max_line_width)[1:-1]
                f.write(f'{test_im} {q_str} {t_str} {inliers} \n')


def generate_precision_recall_plots(file_path, pose_threshold):
    results_dict = np.load(file_path, allow_pickle=True).item()
    out_file_path = os.path.join(os.path.split(file_path)[0], 'pr_')

    all_inliers = []
    all_terr = []
    all_rerr = []
    all_failures = 0

    def plot(prec, rec):
        plt.figure()
        plt.plot(rec, prec, drawstyle='steps-post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(0, 1)
        plt.ylim(0, 1.1)

    # per-scene plots
    for scene, scene_res in results_dict.items():
        terr = np.array([res['abs_t_err'] for res in scene_res.values() if res is not None])
        rerr = np.array([res['abs_r_err'] for res in scene_res.values() if res is not None])
        inliers = np.array([res['inliers'] for res in scene_res.values() if res is not None])
        failures = sum([1 for res in scene_res.values() if res is None])
        all_terr.append(terr)
        all_rerr.append(rerr)
        all_inliers.append(inliers)
        all_failures += failures
        prec, rec, average_precision = precision_recall_pose_error(
            inliers, terr, rerr, failures, pose_threshold)

        plot(prec, rec)
        plt.title(f'Scene {scene}. AP={average_precision:.2f}')
        plt.tight_layout()
        plt.savefig(out_file_path + scene + '.jpg')
        plt.close()

    # whole dataset plot
    terr = np.concatenate(all_terr)
    rerr = np.concatenate(all_rerr)
    inliers = np.concatenate(all_inliers)
    prec, rec, average_precision = precision_recall_pose_error(
        inliers, terr, rerr, failures, pose_threshold)
    plot(prec, rec)
    plt.title(f'Dataset. AP={average_precision:.2f}')
    plt.tight_layout()
    plt.savefig(out_file_path + 'all.jpg')


def eval_pipeline_with_ransac(result_dict, log, ransac_thres, ransac_iter,
                              ransac_miu, pair_type, err_thres, save_res_path=None):
    print('>>>>Evaluate model with Ransac(iter={}, miu={}) Error thres:{})'.format(
        ransac_iter, ransac_miu, err_thres))
    t1 = time.time()
    best_abs_err = None  # TODO: not used for now, remove it in the end
    for thres in ransac_thres:
        avg_err = []
        avg_pass = []
        print('\n>>Ransac threshold:{}'.format(thres))
        loc_results_dict = {}
        for dataset in result_dict:
            start_time = time.time()
            pair_data = result_dict[dataset]['pair_data']
            loc_results_dict[dataset] = {} if save_res_path else None
            if pair_type == 'angess':    # Since angles have been converted to relative poses
                pair_type = 'relapose'
            tested_num, approx_queries, pass_rate, err_res = ransac(
                pair_data, thres, in_iter=ransac_iter, pair_type=pair_type, err_thres=err_thres,
                loc_results=loc_results_dict[dataset])
            avg_err.append(err_res)
            avg_pass.append(pass_rate)
            total_time = time.time() - start_time
            dataset_pr_len = min(10, len(dataset))
            print('Dataset:{dataset} Bad/All:{approx_num}/{tested_num}, Rela:(t{err_res[0]:.2f}deg, r{err_res[1]:.2f}deg) Abs:(t{err_res[2]:.2f}m/{err_res[3]:.2f}deg, r{err_res[4]:.2f}deg) Pass:'.format(
                dataset=dataset[0:dataset_pr_len], approx_num=len(approx_queries), tested_num=tested_num, err_res=err_res) + '/'.join('{:.2f}%'.format(v) for v in pass_rate))

        avg_err = tuple(np.mean(avg_err, axis=0))
        avg_pass = tuple(np.mean(avg_pass, axis=0)) if len(err_thres) > 1 else tuple(avg_pass)
        if best_abs_err is not None:
            if best_abs_err[0] < avg_err[2]:
                best_abs_err = (avg_err[2], avg_err[4])
        else:
            best_abs_err = (avg_err[2], avg_err[4])
        print('Avg: Rela:(t{err_res[0]:.2f}deg, r{err_res[1]:.2f}deg) Abs:(t{err_res[2]:.2f}m/{err_res[3]:.2f}deg, r{err_res[4]:.2f}deg) Pass:'.format(
            err_res=avg_err) + '/'.join('{:.2f}%'.format(v) for v in avg_pass))

        if save_res_path:
            np.save(save_res_path, loc_results_dict)
    time_stamp = 'Ransac testing time: {}s\n'.format(time.time() - t1)
    print(time_stamp)
    return best_abs_err, avg_pass


def eval_pipeline_without_ransac(result_dict, err_thres=(2, 5), log=None, save_res_path=None):
    avg_rela_t_err = []         # Averge relative translation error in angle over datasets
    avg_rela_q_err = []         # Average relative roataion(quternion) error in angle over datasets
    avg_abs_c_dist_err = []     # Averge absolute position error in meter over datasets
    avg_abs_c_ang_err = []      # Averge absolute position error in angle over datasets
    avg_abs_q_err = []          # Averge absolute roataion(quternion) angle error over dataset
    avg_passed = []

    loc_results_dict = {}
    for dataset in result_dict:
        loc_results_dict[dataset] = {} if save_res_path else None
        pair_data = result_dict[dataset]['pair_data']
        failures = result_dict[dataset]['no_pt_pairs']
        print('>>Testing dataset: {}, testing samples: {}, failures {}'.format(
            dataset, len(pair_data), len(failures)))

        # Calculate relative pose error
        rela_t_err, rela_q_err = cal_rela_pose_err(pair_data)
        avg_rela_t_err.append(rela_t_err)
        avg_rela_q_err.append(rela_q_err)

        # Calculate testing pose error with all training images
        abs_c_dist_err, abs_c_ang_err, abs_q_err, passed, average_precision = cal_abs_pose_err_metric(
            pair_data, err_thres, loc_results_dict[dataset])
        avg_abs_c_dist_err.append(abs_c_dist_err)
        avg_abs_c_ang_err.append(abs_c_ang_err)
        avg_abs_q_err.append(abs_q_err)
        avg_passed.append(passed)

        print('rela_err (t{:.2f}deg, r{:.2f}deg) abs err: (t{:.2f}m/{:.2f}deg, r{:.2f}deg), Recall: {}. AP: {:.2f}'.format(rela_t_err,
              rela_q_err, abs_c_dist_err, abs_c_ang_err, abs_q_err, '/'.join('{:.2f}%'.format(v) for v in passed), average_precision))

    if save_res_path:
        np.save(save_res_path, loc_results_dict)

    avg_passed = np.stack(avg_passed).mean(axis=0)
    eval_val = (
        np.mean(avg_rela_t_err),
        np.mean(avg_rela_q_err),
        np.mean(avg_abs_c_dist_err),
        np.mean(avg_abs_c_ang_err),
        np.mean(avg_abs_q_err))
    print('>>avg_rela_err (t{eval_val[0]:.2f}deg, r{eval_val[1]:.2f}deg) avg_abs_err (t{eval_val[2]:.2f}m/{eval_val[3]:.2f}deg, r{eval_val[4]:.2f}deg). Pass:'.format(
        eval_val=eval_val) + '/'.join('{:.2f}%'.format(v) for v in avg_passed))
    return eval_val, avg_passed


def cal_rela_pose_err(pair_data):
    """Calculate relative pose median errors directly over all tested pairs, including:
       - relative translation angle error
       - relative quaternion angle error
    """
    rela_q_err = []
    rela_t_err = []
    for test_im in pair_data:
        test_pair_list = pair_data[test_im]['test_pairs']
        for test_pair in test_pair_list:
            rela_t_err.append(cal_vec_angle_error(
                test_pair.rela_pose_pred.t, test_pair.rela_pose_lbl.t))
            rela_q_err.append(cal_quat_angle_error(
                test_pair.rela_pose_pred.q, test_pair.rela_pose_lbl.q))
    return np.median(rela_t_err), np.median(rela_q_err)


def geometric_median(X, eps=1e-5, axis=0):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y.reshape(1, -1)
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1.reshape(1, -1)

        y = y1


def precision_recall_pose_error(inliers, terr, rerr, failures, pose_threshold):
    """
    Computes Precision/Recall plot for a set of poses given inliers (confidence) and wether the estimated pose error (whatever it may be) is within a threshold.
    Each point in the plot is obtained by choosing a threshold for inliers (i.e. inlier_thr).
    Inputs:
        - inliers [N]
        - terr [N]
        - rerr [N]
        - failures (int)
        - pose_threshold (tuple float)
    Output
        - precision [N]
        - recall [N]
        - average_precision (scalar)
    """

    assert len(inliers) == len(terr) == len(rerr), 'unequal shapes'
    assert len(pose_threshold) == 2, 'invalid pose_threshold'

    tp = (np.array(terr).reshape(-1) <= pose_threshold[0]
          ) * (np.array(rerr).reshape(-1) <= pose_threshold[1])
    return precision_recall(inliers, tp, failures)


def precision_recall_repr_error(inliers, reprerr, failures, repr_threshold):
    """
    Computes Precision/Recall plot for a set of poses given inliers (confidence) and wether the reprojection error is within a threshold (repr_threshold).
    Each point in the plot is obtained by choosing a threshold for inliers (i.e. inlier_thr).
    Inputs:
        - inliers [N]
        - reprerr [N]
        - failures (int)
        - repr_threshold (float)
    Output
        - precision [N]
        - recall [N]
        - average_precision (scalar)
    """
    assert len(inliers) == len(reprerr), 'unequal shapes'

    tp = np.array(reprerr).reshape(-1) < repr_threshold
    return precision_recall(inliers, tp, failures)


def precision_recall(inliers, tp, failures):
    """
    Computes Precision/Recall plot for a set of poses given inliers (confidence) and wether the estimated pose error (whatever it may be) is within a threshold.
    Each point in the plot is obtained by choosing a threshold for inliers (i.e. inlier_thr).
    Recall measures how many images have inliers >= inlier_thr
    Precision measures how many images that have inliers >= inlier_thr have estimated pose error <= pose_threshold (measured by counting tps)
    Where pose_threshold is (trans_thr[m], rot_thr[deg])

    Inputs:
        - inliers [N]
        - terr [N]
        - rerr [N]
        - failures (int)
        - pose_threshold (tuple float)
    Output
        - precision [N]
        - recall [N]
        - average_precision (scalar)
    """

    assert len(inliers) == len(tp), 'unequal shapes'

    # sort by inliers (descending order)
    inliers = np.array(inliers)
    sort_idx = np.argsort(inliers)[::-1]
    inliers = inliers[sort_idx]
    tp = np.array(tp).reshape(-1)[sort_idx]

    # get idxs where inliers change (avoid tied up values)
    distinct_value_indices = np.where(np.diff(inliers))[0]
    threshold_idxs = np.r_[distinct_value_indices, inliers.size - 1]

    # compute prec/recall
    N = inliers.shape[0]
    rec = np.arange(N, dtype=np.float32) + 1
    cum_tp = np.cumsum(tp)
    prec = cum_tp[threshold_idxs] / rec[threshold_idxs]
    rec = rec[threshold_idxs] / (float(N) + float(failures))

    # invert order and ensures (prec=1, rec=0) point
    last_ind = rec.searchsorted(rec[-1])
    sl = slice(last_ind, None, -1)
    prec = np.r_[prec[sl], 1]
    rec = np.r_[rec[sl], 0]

    # compute average precision (AUC) as the weighted average of precisions
    average_precision = np.abs(np.sum(np.diff(rec) * np.array(prec)[:-1]))

    return prec, rec, average_precision


def cal_abs_pose_err_metric(pair_data, err_thres=(2, 5), loc_results=None):
    """Calculate absolute pose median errors directly (No RANSAC) over all tested pairs WITHOUT triangulation (assuming
         metric relative pose), including:
       - absolute positional distance error in meter
       - absolute positional angle error
       - absolute rotational angle error
       - Precision, Recall and Average Precision (AP)
    """
    abs_c_dist_err = []
    abs_c_ang_err = []
    abs_q_err = []
    inliers = []
    passed = [0] * len(err_thres)
    failures = 0
    for test_im in pair_data:
        test_abs_pose = pair_data[test_im]['test_abs_pose']
        test_pair_list = pair_data[test_im]['test_pairs']

        # no pairs means failure
        if len(test_pair_list) == 0:
            failures += 1
            loc_results[test_im] = None  # indicates failure
            continue

        # Estimate absolute pose of test image
        abs_q_pred_list = []
        train_abs_c_list = []
        abs_c_pred_list = []
        inliers_list = []
        for test_pair in test_pair_list:
            abs_q_pred_list.append(test_pair.abs_q_pred)
            train_abs_c_list.append(test_pair.train_abs_pose.c)
            abs_c_pred_list.append(test_pair.abs_c_pred)
            inliers_list.append(test_pair.inliers)
        abs_c_pred = geometric_median(np.vstack(abs_c_pred_list), axis=0)
        cerr = np.linalg.norm(test_abs_pose.c - abs_c_pred, axis=1)
        abs_c_dist_err.append(cerr)
        train_abs_c = np.vstack(train_abs_c_list)
        # Angle between abs_pose label and prediction, with train image as reference point
        abs_c_ang_err.append(np.median(cal_vec_angle_error(
            test_abs_pose.c - train_abs_c, abs_c_pred - train_abs_c)))
        inliers.append(inliers_list[0])  # assumes a single keyframe

        # Use Scipy Rotation Averaging (L2 mean)
        abs_r_pred = np.stack([quat2mat(q) for q in abs_q_pred_list])
        abs_r_pred = Rotation.from_matrix(abs_r_pred).mean().as_matrix()
        abs_q_pred = mat2quat(abs_r_pred)
        # Traditional q averaging
        # abs_q_pred = np.mean(np.vstack(abs_q_pred_list), axis=0)
        qerr = cal_quat_angle_error(test_abs_pose.q, abs_q_pred)
        abs_q_err.append(qerr)

        for i_e, err_t in enumerate(err_thres):
            if cerr < err_t[0] and qerr < err_t[1]:
                passed[i_e] += 1

        # Save results for further analysis
        if loc_results is not None:
            res = {}
            res['abs_pose_lbl'] = test_abs_pose
            res['abs_pose_pred'] = AbsPose(abs_q_pred.reshape(-1), abs_c_pred.reshape(-1))
            res['abs_t_err'] = cerr.item()
            res['abs_r_err'] = qerr.item()
            res['inliers'] = inliers_list[0]  # assumes a single keyframe
            loc_results[test_im] = res

    prec, rec, average_precision = precision_recall_pose_error(
        inliers, abs_c_dist_err, abs_q_err, failures, pose_threshold=err_thres[1])
    passed = np.array(passed)
    return np.median(abs_c_dist_err), np.median(abs_c_ang_err), np.median(abs_q_err), 100.0 * passed / len(pair_data), average_precision

# uses triangulation, since assumes non-metric relative pose


def cal_abs_pose_err(pair_data, err_thres=(2, 5)):
    """Calculate absolute pose median errors directly (No RANSAC) over all tested pairs, including:
       - absolute positional distance error in meter
       - absolute positional angle error
       - absolute rotational angle error
    """
    abs_c_dist_err = []
    abs_c_ang_err = []
    abs_q_err = []
    passed = 0
    for test_im in pair_data:
        test_abs_pose = pair_data[test_im]['test_abs_pose']
        test_pair_list = pair_data[test_im]['test_pairs']
        k = len(test_pair_list)
        if k == 1:
            continue    # Triangulation requires at least 2 training images

        # Estimate absolute pose of test image
        abs_q_pred_list = []
        correspondence = []
        train_abs_c_list = []
        for test_pair in test_pair_list:
            correspondence.append((test_pair.x_te, test_pair.train_abs_pose.p))
            abs_q_pred_list.append(test_pair.abs_q_pred)
            train_abs_c_list.append(test_pair.train_abs_pose.c)
        abs_c_pred = triangulate_multi_views(correspondence)
        cerr = np.linalg.norm(test_abs_pose.c - abs_c_pred)
        abs_c_dist_err.append(cerr)
        train_abs_c = np.vstack(train_abs_c_list)
        # Angle between abs_pose label and prediction, with train image as reference point
        abs_c_ang_err.append(np.mean(cal_vec_angle_error(
            test_abs_pose.c - train_abs_c, abs_c_pred - train_abs_c)))
        abs_q_pred = np.mean(np.vstack(abs_q_pred_list), axis=0)
        qerr = cal_quat_angle_error(test_abs_pose.q, abs_q_pred)
        abs_q_err.append(qerr)

        if cerr < err_thres[0] and qerr < err_thres[1]:
            passed += 1

    return np.median(abs_c_dist_err), np.median(abs_c_ang_err), np.median(abs_q_err), 100.0*passed/len(abs_q_err)


########################
#### RANSAC METHODS#####
########################
def ransac(pair_data, inlier_thres, thres_multiplier=1.414, in_iter=10, pair_type='ess',
           err_thres=[(0.25, 2), (0.5, 5), (5, 10)], loc_results=None):
    abs_c_dist_err = []
    abs_c_ang_err = []
    abs_q_err = []
    rela_t_err = []
    rela_q_err = []
    passed = [0 for thres in err_thres]
    approx_queries = []
    for test_im in pair_data:
        test_abs_pose = pair_data[test_im]['test_abs_pose']
        test_pair_list = pair_data[test_im]['test_pairs']
        num_pair = len(test_pair_list)

        if num_pair == 0:
            # There's no pair predictes valid essentials
            # Manually set big errors, median error should be robust to such outliers
            cerr = 1000
            qerr = 180
            abs_c_dist_err.append(cerr)
            abs_c_ang_err.append(qerr)
            abs_q_err.append(qerr)
            rela_t_err.append(qerr)
            rela_q_err.append(qerr)
            loc_results[test_im] = None  # indicates failure
        else:
            # Run Ransac algorithm
            inlier_best = []
            abs_pose_best = None
            approximated = False
            # Check all possible pairs as minimal inlier samples
            inlier_min_samples = list(itertools.combinations(range(num_pair), 2))
            for inlier_min in inlier_min_samples:
                # The initial hypothesis in different manner according to the structure(type) of test pair
                if pair_type == 'ess':
                    pair0, pair1 = test_pair_list[inlier_min[0]], test_pair_list[inlier_min[1]]
                    err_min = 1000
                    id0, id1 = -1, -1

                    # Determine the two rotations for the pair by choosing the combination with smallest angle error
                    for i in range(2):
                        for j in range(2):
                            err = cal_quat_angle_error(pair0.abs_q_pred[i], pair1.abs_q_pred[j])
                            if err < err_min:
                                err_min = err
                                id0, id1 = i, j
                    # Use the average quaternion over 2 pairs
                    abs_q_hypo = np.mean(
                        np.vstack([pair0.abs_q_pred[id0], pair1.abs_q_pred[id1]]), axis=0)
                    x0, x1 = pair0.x_te[id0], pair1.x_te[id1]
                    p0, p1 = pair0.train_abs_pose.p, pair1.train_abs_pose.p
                    abs_c_hypo = triangulate_two_views(x0, p0, x1, p1)
                    abs_pose_hypo = AbsPose(abs_q_hypo, abs_c_hypo)
                if pair_type == 'relapose':
                    abs_pose_hypo = estimate_model(test_pair_list, inlier_min, pair_type)
                inlier_hypo = find_inliers(abs_pose_hypo, test_pair_list,
                                           inlier_thres, pair_type=pair_type)

                # Perform local optimation step if this hypo has so far most inliers
                if len(inlier_hypo) >= 2 and len(inlier_hypo) > len(inlier_best):
                    inlier_best = inlier_hypo
                    abs_pose_best = abs_pose_hypo

                    # local optimisation
                    inlier_local_best, pose_local_best = local_optimisation(
                        test_pair_list, abs_pose_best, thres_multiplier, inlier_thres, in_iter, pair_type)
                    if len(inlier_local_best) > len(inlier_best):
                        inlier_best = inlier_local_best
                        abs_pose_best = pose_local_best

            if abs_pose_best is None or len(inlier_best) == 0:
                # In this case, either the pair has bad confidence
                # Or there's one training pair for this query
                # Use pose of training to approximate the pose
                inlier_id = 0
                pair = test_pair_list[inlier_id]
                abs_pose_best = pair.train_abs_pose
                inlier_best = [inlier_id]
                approx_queries.append(test_im)
                approximated = True

            if pair_type == 'ess':
                # Identify the correct relative translation based on the best hypothesis
                find_inliers(abs_pose_best, test_pair_list, inlier_thres,
                             pair_type=pair_type, update_trans=True)  # Identify t and update

            # Calculate relative error with best inlier set
            train_abs_c_list = []
            t_err = []
            q_err = []
            cumulative_correspondences_inliers = 0
            for i in inlier_best:
                test_pair = test_pair_list[i]
                train_abs_c_list.append(test_pair.train_abs_pose.c)
                if pair_type == 'ess':
                    t_err.append(cal_vec_angle_error(test_pair.t, test_pair.rela_pose_lbl.t))
                    q_err.append(
                        cal_quat_angle_error(
                            test_pair.get_rela_q(),
                            test_pair.rela_pose_lbl.q))
                if pair_type == 'relapose':
                    t_err.append(
                        cal_vec_angle_error(
                            test_pair.rela_pose_pred.t, test_pair.rela_pose_lbl.t))
                    q_err.append(
                        cal_quat_angle_error(
                            test_pair.rela_pose_pred.q, test_pair.rela_pose_lbl.q))
                cumulative_correspondences_inliers += test_pair.inliers
            rela_t_err.append(np.mean(t_err))
            rela_q_err.append(np.mean(q_err))

            # Calculate absolute error
            if len(train_abs_c_list) > 1:
                train_abs_c = np.vstack(train_abs_c_list)
            else:
                train_abs_c = train_abs_c_list[0]
                train_abs_c.reshape((1, 3))
            abs_pose_pred = abs_pose_best
            cerr = np.linalg.norm(test_abs_pose.c - abs_pose_pred.c)
            abs_c_dist_err.append(cerr)

            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                try:
                    if approximated:
                        # Prevent zero division
                        abs_c_ang_err.append(0.0)
                    else:
                        # Angle between abs_pose label and prediction, with train image as reference point
                        abs_c_ang_err.append(
                            np.mean(
                                cal_vec_angle_error(
                                    test_abs_pose.c - train_abs_c, abs_pose_pred.c - train_abs_c)))
                except Warning:
                    print('Warning catched during abs angle error calculation')
                    print('Test im {}, num_pair {}'.format(test_im, len(test_pair_list)))

            qerr = cal_quat_angle_error(test_abs_pose.q, abs_pose_pred.q).squeeze()
            abs_q_err.append(qerr)

            # Save results for further analysis
            if loc_results is not None:  # TODO: Separate process for error analysis
                res = {}
                res['abs_pose_lbl'] = test_abs_pose
                res['abs_pose_pred'] = abs_pose_best
                res['relv_pose_list'] = test_pair_list
                # the confidence is the sum of corr. inliers of each inlier pose pair
                res['inliers'] = cumulative_correspondences_inliers
                res['approximated'] = approximated
                res['abs_t_err'] = cerr.item()
                res['abs_r_err'] = qerr.item()
                loc_results[test_im] = res

        # DSAC eval criterion: cerr < thres (m) & qerr < thres (deg)
        for i, thres in enumerate(err_thres):
            cerr_thres, qerr_thres = thres
            if cerr < cerr_thres and qerr < qerr_thres:
                passed[i] += 1
    num_tested = len(abs_c_dist_err)
    pass_rate = [100.0 * count / num_tested for count in passed]
    return num_tested, approx_queries, pass_rate, (np.median(rela_t_err),
                                                   np.median(rela_q_err),
                                                   np.median(abs_c_dist_err),
                                                   np.median(abs_c_ang_err),
                                                   np.median(abs_q_err))


def local_optimisation(test_pair_list, abs_pose_best, thres_multiplier, thres, in_iter, pair_type):
    # Re-evaluate model and inliers with threshold multiplied by thres_multiplier
    inlier_mult = find_inliers(abs_pose_best, test_pair_list,
                               thres_multiplier*thres, pair_type=pair_type)
    abs_pose_mult = estimate_model(test_pair_list, inlier_mult, pair_type)
    inlier_base = find_inliers(abs_pose_mult, test_pair_list, thres, pair_type=pair_type)

    # Evaluate model from subsample of inlier_base
    inlier_base_sample = list(inlier_base)
    all_abs_poses = [abs_pose_best, abs_pose_mult]
    num_inlier_subsample = min(14, int(len(inlier_base)/2))
    if num_inlier_subsample > 2:
        for i in range(in_iter):
            np.random.shuffle(inlier_base_sample)
            inlier_subsample = inlier_base_sample[:num_inlier_subsample]
            abs_pose_subsample = estimate_model(test_pair_list, inlier_subsample, pair_type)
            all_abs_poses.append(abs_pose_subsample)

    # Identify the best model
    inlier_local_best = []
    pose_local_best = None
    for abs_pose in all_abs_poses:
        inlier_ = find_inliers(abs_pose, test_pair_list, thres, pair_type=pair_type)
        if len(inlier_) > len(inlier_local_best):
            inlier_local_best = inlier_
            pose_local_best = abs_pose
    return inlier_local_best, pose_local_best


def find_inliers(hypo_abs_pose, test_pair_list, thres, pair_type='ess', update_trans=False):
    """
    Find inliers from the full sample set based on the hypothesised absolute pose of the test image.
    The train image is counted as an inlier if the estimated translation angle error between 
    it and the test image is within the threshold.
    Args:
        - hypo_abs_pose: AbasPose object represents the absolute pose hypothesis of the test image
        - test_pair_list: the full set of testing pairs, i.e., a list of RelaPosePair/EssPair objects
        - thres: error threshold to filter the outliers
        - pair_type: specifies the type pair objects: 'relapose'->RelaPosePair, 'ess'->EssPair
        - update_trans: specifies whether to determine the correct sign of the relative translation, 
          this is only used when the pair_type is 'ess'. And the update is performed after the best absolute hypothese
          is found and the sign giving smaller angle difference from the hypothesed translation is picked.
    Return:
        - the inlier indices of test pairs from the test_pair_list
    """
    inliers = []
    k = len(test_pair_list)
    for i in range(k):
        test_pair = test_pair_list[i]

        # Estimate relative translation (from query to train)based on the hypothesis
        train_abs_pose = test_pair.train_abs_pose
        rela_t_est = train_abs_pose.r.dot(hypo_abs_pose.c - train_abs_pose.c)

        # Optimal reltaive pose predicted by the network
        if pair_type == 'ess':
            # Identify the correct rotation of each pair with hypothesis
            err0 = cal_quat_angle_error(hypo_abs_pose.q, test_pair.abs_q_pred[0])
            err1 = cal_quat_angle_error(hypo_abs_pose.q, test_pair.abs_q_pred[1])
            rid = np.argmin([err0, err1])
            test_pair.set_rid(rid)
            rela_r_opt = test_pair.R[rid]
            rela_t_opt = test_pair.t
        if pair_type == 'relapose':
            rela_r_opt = test_pair.rela_pose_pred.r
            rela_t_opt = test_pair.rela_pose_pred.t
        t_est = rela_t_est
        t_opt = - rela_r_opt.T.dot(rela_t_opt)  # reversed direction, i.e., from query to train im

        err = np.inf
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=RuntimeWarning)
            try:
                if np.linalg.norm(t_est) == 0.0:
                    # Training and testing image has same position
                    err = 0.0  # Check whether this is appropriate
                else:
                    # Calculate translation angle error and locate inliers by threshoding
                    err = cal_vec_angle_error(t_est, t_opt)
                    if pair_type == 'ess':
                        # Identify the correct translation that giving smaller error
                        t_opt_ = - t_opt
                        err_ = cal_vec_angle_error(t_est, t_opt_)
                        if err_ < err:
                            err = err_
                            if update_trans:
                                test_pair.set_opposite_trans_pred()  # Update to the correct translation in the pair data
            except Warning:
                print('Warning catched during find inlier calculation')
                print('Test im {}, Train {}'.format(test_pair.test_im, test_pair.train_im))

        if err < thres:
            inliers.append(i)
    return inliers


def estimate_model(test_pair_list, inliers, pair_type):
    """Estimate absolute pose of test image 
    Args:
        - test_pair_list: the full set of testing pairs, i.e., a list of RelaPosePair/EssPair objects
        - inliers: list of indices pointing to the inlier test pairs
        - pair_type: specifies the type pair objects: 'relapose'->RelaPosePair, 'ess'->EssPair
    Return:
        - an AbasPose object representing the absolute pose prediction with the given inliers
    """
    abs_q_pred_list = []
    correspondence = []
    for i in inliers:
        test_pair = test_pair_list[i]
        if pair_type == 'ess':
            rid = test_pair.rid
            correspondence.append((test_pair.x_te[rid], test_pair.train_abs_pose.p))
            abs_q_pred_list.append(test_pair.abs_q_pred[rid])
        if pair_type == 'relapose':
            correspondence.append((test_pair.x_te, test_pair.train_abs_pose.p))
            abs_q_pred_list.append(test_pair.abs_q_pred)
    abs_c_pred = triangulate_multi_views(correspondence)
    abs_q_pred = np.mean(np.vstack(abs_q_pred_list), axis=0)
    return AbsPose(abs_q_pred, abs_c_pred)

#############################
####EPIPOLAR CALCULATION#####
#############################


def triangulate_two_views(x1, p1, x2, p2):
    """Triangulate a 3d point from 2 views 
    Args:
        - x1: point correspondence of target 3d point in 1st view
        - p1: projection matrix of 1st view i.e. [R1|t1]
        - x2: point correspondence  of target 3d point in 2nd view
        - p1: projection matrix of 2nd view i.e. [R2|t2]
    Return:
        - X: triangulated 3d point, shape (3,)
    """
    A_rows = []
    A_rows.append(np.expand_dims(x1[0]*p1[2, :] - p1[0, :], axis=0))
    A_rows.append(np.expand_dims(x1[1]*p1[2, :] - p1[1, :], axis=0))
    A_rows.append(np.expand_dims(x2[0]*p2[2, :] - p2[0, :], axis=0))
    A_rows.append(np.expand_dims(x2[1]*p2[2, :] - p2[1, :], axis=0))
    A = np.concatenate(A_rows, axis=0)

    # Find null space of A
    u, s, vh = np.linalg.svd(A)
    X = vh[-1, :]    # Last column of v
    X = X[:3] / X[3]
    return X


def triangulate_multi_views(correspondence):
    """Triangulate a 3d point from multiple views
    Args:
        - correspondence = list of (xi, pi) where
            xi: 2d point correspondence of target 3d point in i-th view
            pi: projection matrix of i-th view i.e. [Ri|ti]
    Return:
        - X: triangulated 3d point, shape (3,)
    """
    A_rows = []
    for (xi, pi) in correspondence:
        A_rows.append(np.expand_dims(xi[0]*pi[2, :] - pi[0, :], axis=0))
        A_rows.append(np.expand_dims(xi[1]*pi[2, :] - pi[1, :], axis=0))
    A = np.concatenate(A_rows, axis=0)

    # Find null space of A
    u, s, vh = np.linalg.svd(A)
    X = vh[-1, :]    # Last column of v
    X = X[:3] / X[3]
    return X


def compose_projection_matrix(R, t):
    """Construct projection matrix 
    Args:
        - R: rotation matrix, size (3,3);
        - t: translation vector, size (3,);
    Return:
        - projection matrix [R|t], size (3,4)
    """
    return np.hstack([R, np.expand_dims(t, axis=1)])


def hat(vec):
    """Skew operator
    Args:
        - vec: vector of size (3,) to be transformed;
    Return: 
        - skew-symmetric matrix of size (3, 3)
    """
    [a1, a2, a3] = list(vec)
    skew = np.array([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])
    return skew


def project_onto_essential_space(F):
    u, s, vh = np.linalg.svd(F)
    a = (s[0] + s[1]) / 2
    s_ = np.array([a, a, 0])
    E = u.dot(np.diag(s_)).dot(vh)
    return E


def essential_matrix_from_pose(R, t):
    """Calculate essential matrix
    Args:
        - R: rotation matrix, size (3,3);
        - t: translation vector, size (3,);
    Return:
        - essential matrix, size (3,3)
    """
    t = t/np.linalg.norm(t)     # force translation to be unit length
    t_skew = hat(t)
    E = t_skew.dot(R)
    return E.astype(np.float32)

# Will give worse results than the opencv implementation for some cases.
# def decompose_essential_matrix(E):
#     """Extract possible pose from essential matrix
#     Args:
#         - E: essential matrix, shape(3,3)
#     Return:
#         - t: one possible translation, the other is -t
#         - R1, R2: possible rotation
#     """
#     u, s, vh = np.linalg.svd(E)
#     if np.linalg.det(u) < 0 or np.linalg.det(vh) < 0:
#         u,s,vh = np.linalg.svd(-E)
#     t = u[:,2]
#     w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
#     R1 = u.dot(w).dot(vh)
#     R2 = u.dot(w.T).dot(vh)
#     return (t,R1,R2)


def decompose_essential_matrix(E):
    """Extract possible pose from essential matrix(opencv version)
    Args:
        - E: essential matrix, shape(3,3)
    Return:
        - t: one possible translation, the other is -t
        - R1, R2: possible rotation
    """
    u, s, vh = np.linalg.svd(E)
    if np.linalg.det(u) < 0:
        u = -u
    if np.linalg.det(vh) < 0:
        vh = -vh
    t = u[:, 2]
    w = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    R1 = u.dot(w).dot(vh)
    R2 = u.dot(w.T).dot(vh)
    return (t, R1, R2)

#############################
####DATA WRAPPER CLASSES#####
#############################


class AbsPose:
    def __init__(self, q, c, init_proj=True):
        """Define an absolute camera pose with respect to the global coordinates
        Args:
            - init_proj: whether initialize projection matrix for this instance
        Attributes:
            - c: absolute position of the camera in global system, shape (3,)
            - q: absolute orientation (in quaternion) of the camera in global system, shape (4,)
            - r: absolute orientation (in rotation) of the camera in global system, shape (3, 3)
            - t: translation vector of the pose 
            - p: the projection matrix that transforms a point in global coordinates to this camera's local coordinate
        """
        assert len(q.shape) == 1
        assert q.shape[0] == 4
        assert len(c.shape) == 1
        assert c.shape[0] == 3

        self.q = q
        self.r = quat2mat(self.q)
        self.c = c
        self.t = -self.r.dot(self.c)
        if init_proj:
            self.p = compose_projection_matrix(self.r, self.t)


class RelaPose:
    def __init__(self, q, t):
        """Define a relaitve camera pose
        Attributes:
            - q: relative orientation (in quaternion), shape (4,)
            - r: relative orientation (in rotation), shape (3, 3)
            - t: relative translation vector of the pose 
        """
        assert len(q.shape) == 1
        assert q.shape[0] == 4
        assert len(t.shape) == 1
        assert t.shape[0] == 3

        self.q = q
        self.r = quat2mat(self.q)
        self.t = t


class RelaPosePair:
    '''This class structures necessary information related to a testing pair for the relative pose regression models'''

    def __init__(self, test_im, train_abs_pose, rela_pose_lbl, rela_pose_pred, sim):
        """Initialize the relative pose data information
        Attributes:
            - test_im: string, the name of the test_im
            - train_abs_pose: AbsPose object, the absolute pose ground truth of the train_im 
            - rela_pose_lbl: RelaPose object, relative pose ground truth        
            - rela_pose_pred: RelaPose object, predicted relative pose by the network
            - x_te : the 2d point correspondence of test_im in this train_im 
            - abs_r_pred : the predicted absolute rotation (SO(3) matrix) of test_im by this train_im		
            - abs_q_pred : the predicted absolute rotation in quaternion
            - abs_c_pred: the predicted absolute camera position (c in R^3) of test_im by this train_im (assume metric relpose)
            - sim: VLAD feature similarity between image pairs
        """
        self.test_im = test_im
        self.train_abs_pose = train_abs_pose
        self.rela_pose_lbl = rela_pose_lbl
        self.rela_pose_pred = rela_pose_pred
        x_te = - self.rela_pose_pred.r.T.dot(self.rela_pose_pred.t)
        self.x_te = x_te[:2] / (x_te[2] if x_te[2] != 0 else 1)
        self.abs_r_pred = self.rela_pose_pred.r.dot(self.train_abs_pose.r)
        self.abs_q_pred = mat2quat(self.abs_r_pred)
        self.abs_c_pred = train_abs_pose.c - self.train_abs_pose.r.T @ self.rela_pose_pred.r.T @ self.rela_pose_pred.t
        self.sim = sim


class EssPair:
    '''This class structures necessary information related to a testing pair for the essential matrix regression models'''

    def __init__(self, test_im, train_im, train_abs_pose, rela_pose_lbl, t, R0, R1):
        """Initialize the relative pose data information
        Attributes:
            - test_im: string, the name of the test_im
            - train_abs_pose: AbsPose object, the absolute pose ground truth of the train_im 
            - rela_pose_lbl: RelaPose object, relative pose ground truth        
            - t: relative translation extracted from an essential matrix, undetermined up to a sign at intialize time. 
                 The sign will be identified in RANSAC using set_opposite_trans_pred()
            - R0, R1: two possible rotation matrices extracted from an essential matrix
            - rid: the index of the correct rotation, either 0 or 1. The rid is set during RANSAC with set_rid()
        The followings are calculated correspondingly using R0 and R1
            - x_te : the two possible 2d point correspondences of test_im in this train_im  
            - abs_r_pred : the two possible predicted absolute rotations in SO(3) matrices of test_im by this train_im		
            - abs_q_pred : the two possible predicted absolute rotations in quaternion
        """
        assert len(t.shape) == 1
        assert t.shape[0] == 3
        assert len(R0.shape) == len(R1.shape) == 2
        assert R0.shape == R1.shape == (3, 3)

        self.train_im = train_im
        self.test_im = test_im
        self.train_abs_pose = train_abs_pose
        self.rela_pose_lbl = rela_pose_lbl
        self.rela_pose_pred = None
        self.t = t
        self.R = [R0, R1]
        self.abs_r_pred = []
        self.abs_q_pred = []
        self.x_te = []
        for i in range(2):
            R = self.R[i]
            x_te = - R.T.dot(self.t)
            if x_te[2] == 0:
                self.x_te.append(np.array([np.inf, np.inf]))
            else:
                self.x_te.append(x_te[:2] / x_te[2])
            self.abs_r_pred.append(R.dot(self.train_abs_pose.r))   # r_t1 = R*r_tr
            self.abs_q_pred.append(mat2quat(self.abs_r_pred[i]))

    def set_rid(self, rid):
        self.rid = rid

    def set_opposite_trans_pred(self):
        self.t = - self.t

    def get_rela_q(self):
        return mat2quat(self.R[self.rid])

    def is_invalid(self):
        return np.any(np.isinf(self.x_te))
