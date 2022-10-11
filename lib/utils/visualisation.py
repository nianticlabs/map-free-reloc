import os

import cv2
import numpy as np
from lib.utils.localize import cal_vec_angle_error, cal_quat_angle_error


def save_video(save_res_path, dataloader, odir):
    """Generate a video per sequence with per frame metrics."""

    from vidgear.gears import WriteGear

    def save_video_gear(odir, old_scene, frames):
        video_writer = WriteGear(
            output_filename=f'{odir / old_scene}.mp4', custom_ffmpeg=os.getenv('FFMPEG_PATH'))
        if not video_writer._WriteGear__ffmpeg:
            print('Could not find ffmpeg path in the system. If available, set ffmpeg path in env. var. FFMPEG_PATH')

        # sort frames by filename and write to disk
        for k, frame in sorted(frames.items(), key=lambda item: item[0]):
            video_writer.write(frame)
        video_writer.close()
        return

    results_dict = np.load(save_res_path, allow_pickle=True).item()
    old_scene = None
    scenes = []
    write_frames = {}

    for data in dataloader:
        scene = data['scene_id'][0]
        train_img_path, test_img_path = data['pair_names'][0][0], data['pair_names'][1][0]

        if scene not in scenes and old_scene is not None:
            save_video_gear(odir, old_scene, write_frames)
            write_frames = {}
            scenes.append(scene)

        # get performance metrics
        try:
            abs_pose_lbl = results_dict[scene][test_img_path]['abs_pose_lbl']
            abs_pose_pred = results_dict[scene][test_img_path]['abs_pose_pred']
            r_err = cal_quat_angle_error(abs_pose_lbl.q, abs_pose_pred.q).item()
            t_ang_err = cal_vec_angle_error(abs_pose_lbl.t, abs_pose_pred.t).item()
            t_err = np.linalg.norm(abs_pose_lbl.c - abs_pose_pred.c).item()
        except:
            r_err = float('inf')
            t_err = float('inf')
            t_ang_err = float('inf')

        # convert frames (pytorch -> OCV)
        c0 = (data['image0'].squeeze(0).permute(1, 2, 0)
              * 255).detach().cpu().numpy().astype(np.uint8)
        c0 = c0[:, :, ::-1]
        c1 = (data['image1'].squeeze(0).permute(1, 2, 0)
              * 255).detach().cpu().numpy().astype(np.uint8)
        c1 = c1[:, :, ::-1]
        frame = np.concatenate((c0, c1), axis=1).copy()

        # write metrics
        text = f'R_err: {r_err:.1f}deg. t_ang_err: {t_ang_err:.1f}deg. t_err: {t_err:.2f}m'
        font_size = 1 if c0.shape[0] > 500 else 0.5
        tx = 100 if c0.shape[0] > 500 else 10
        ty = c0.shape[0] - 30
        color = (0, 255, 0) if r_err <= 5 and t_err <= 0.25 else (0, 0, 255)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX,
                    font_size, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX,
                    font_size, color, 1, cv2.LINE_AA)

        # resize
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        # save in memory (need to order before saving)
        write_frames[test_img_path] = frame
        old_scene = scene

    # last sequence
    save_video_gear(odir, old_scene, write_frames)
    return
