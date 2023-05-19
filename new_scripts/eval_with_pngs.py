# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import os
import argparse
import fnmatch
import cv2
import numpy as np
import pdb
from visualize_panoptics import plot_depth
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS TensorFlow implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--pred_path',           type=str,   help='path to the prediction results in png', required=True)
parser.add_argument('--gt_path',             type=str,   help='root path to the groundtruth data', required=False)
parser.add_argument('--dataset',             type=str,   help='dataset to test on, nyu or kitti', default='nyu')
parser.add_argument('--eigen_crop',                      help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                       help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--min_depth_eval',      type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',      type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--do_kb_crop',                      help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--is_bdd100k',                      help='if set, it used full scale bdd100k images', action='store_true')
parser.add_argument('--eval_center_only',                help='if set, it only uses center of image', action='store_true')
parser.add_argument('--use_virtual_depth',               help='if set, use virtual depth', action='store_true')
parser.add_argument('--do_kb_crop_on_cityscapes',        help='if set, use virtual depth', action='store_true')

args = parser.parse_args()


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


def test():
    global gt_depths, missing_ids, pred_filenames
    gt_depths = []
    missing_ids = set()
    pred_filenames = []

    for root, dirnames, filenames in os.walk(args.pred_path):
        for pred_filename in fnmatch.filter(filenames, '*.npy'):
            if 'cmap' in pred_filename or 'gt' in pred_filename:
                continue
            dirname = root.replace(args.pred_path, '')
            pred_filenames.append(os.path.join(dirname, pred_filename))

    num_test_samples = len(pred_filenames)

    pred_depths = []

    for i in range(num_test_samples):
        pred_depth_path = os.path.join(args.pred_path, pred_filenames[i])
        pred_depth = np.load(pred_depth_path)
        if pred_depth is None:
            print('Missing: %s ' % pred_depth_path)
            missing_ids.add(i)
            continue

        if args.dataset == 'nyu':
            pred_depth = pred_depth.astype(np.float32) / 1000.0
        else:
            pred_depth = pred_depth.astype(np.float32)

        pred_depths.append(pred_depth)

    print('Raw png files reading done')
    print('Evaluating {} files'.format(len(pred_depths)))

    for t_id in range(num_test_samples):
        gt_depth_path = os.path.join(args.gt_path, pred_filenames[t_id])
        depth = np.load(gt_depth_path)
        if depth is None:
            print('Missing: %s ' % gt_depth_path)
            missing_ids.add(t_id)
            continue

        depth = depth.astype(np.float32)
        gt_depths.append(depth)

    print('GT files reading done')
    print('{} GT files missing'.format(len(missing_ids)))

    print('Computing errors')
    eval(pred_depths)

    print('Done.')


def eval(pred_depths):

    num_samples = len(pred_depths)
    pred_depths_valid = []

    i = 0
    for t_id in range(num_samples):
        if t_id in missing_ids:
            continue

        pred_depths_valid.append(pred_depths[t_id])

    num_samples = num_samples - len(missing_ids)

    silog = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1 = np.zeros(num_samples, np.float32)
    d2 = np.zeros(num_samples, np.float32)
    d3 = np.zeros(num_samples, np.float32)
    
    for i in range(num_samples):

        gt_depth = gt_depths[i]
        pred_depth = pred_depths_valid[i]

        if args.dataset == 'cityscapes':
            height, width = pred_depth.shape
            top_margin = int((height - 352))
            left_margin = int((width - 1216) / 2)
            pred_depth = pred_depth[top_margin:top_margin + 352, left_margin:left_margin + 1216]

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        if args.use_virtual_depth:
            pred_depth = pred_depth * 715.0873 / 1020.0 #* 720 / 352

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        gt_depth[np.isinf(gt_depth)] = 0
        gt_depth[np.isnan(gt_depth)] = 0

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
        
        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti' or "cityscapes":
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                else:
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

            if args.eval_center_only:
                height, width = gt_depth.shape
                top_margin = int((height - 352) / 2)
                left_margin = int((width - 1216) / 2)
                center_mask = np.zeros(valid_mask.shape)
                center_mask[top_margin:top_margin + 352, left_margin:left_margin + 1216] = np.ones((352, 1216))
                valid_mask = np.logical_and(valid_mask, center_mask)
            
        # range_mask = np.zeros(valid_mask.shape)
        # range_mask[np.where(pred_depth > 0)] = 1.0
        # valid_mask = np.logical_and(valid_mask, range_mask)
        silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
    line1 = "{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10')
    line2 = "{:7.3f}, {:7.3f}, {:7.3f}, {:7.5f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
        np.nanmean(d1), np.nanmean(d2), np.nanmean(d3),
        np.nanmean(abs_rel), np.nanmean(sq_rel), np.nanmean(rms), np.nanmean(log_rms), np.nanmean(silog), np.nanmean(log10))
    print(line1)
    print(line2)
    pdb.set_trace()
    save_path = pathlib.PurePath(pathlib.Path(args.pred_path)).parent
    
    text_file = open(os.path.join(save_path, "eval.txt"), "w")
    text_file.write(line1 + "\n")
    text_file.write(line2)
    text_file.close()

    return silog, log10, abs_rel, sq_rel, rms, log_rms, d1, d2, d3


def main():
    test()


if __name__ == '__main__':
    main()



