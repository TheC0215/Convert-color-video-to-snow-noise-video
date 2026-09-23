# -*- coding: utf-8 -*-
"""
run.py — 两步流水线：彩色视频 → 深度视频 → 雪花噪声视频
========================================================================
第 1 步：使用 Video-Depth-Anything 处理 Videos Pending Processing 中的视频，
        生成深度可视化视频（_vis.mp4，以及源视频 _src.mp4）。
第 2 步：基于深度视频生成 "Lost in the Static" 风格雪花噪声视频（_vis_static.mp4）
        原理：所有像素均匀随机噪声（单帧看不出结构），远处雪花完全静止、
        近处雪花快速闪烁，人眼通过运动视差感知深度。

用法：
    D:\anaconda3\envs\vda\python.exe run.py --encoder vits
        # 默认处理 Videos Pending Processing\Tokyo-Walk_rgb.mp4
        # 依次生成 _src.mp4 → _vis.mp4 → _vis_static.mp4
    D:\anaconda3\envs\vda\python.exe run.py --input_video <视频或文件夹> --encoder vits
    D:\anaconda3\envs\vda\python.exe run.py --encoder vitl --far_th 0.3 --near_th 0.6
    D:\anaconda3\envs\vda\python.exe run.py --encoder vits --no_snow   # 只做深度，不做雪花
"""
import argparse
import glob
import os
import sys

# 保证中文输出在 Windows 控制台正常
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import cv2
import torch

# 路径设定（run.py 位于 "Convert color video to snow noise video" 根目录）
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(ROOT, 'Video-Depth-Anything-main')          # 项目目录
VIDEO_DIR = os.path.join(ROOT, 'Videos Pending Processing')        # 待处理视频目录
OUTPUT_DIR = VIDEO_DIR                                             # 输出到视频所在文件夹
DEFAULT_VIDEO = os.path.join(VIDEO_DIR, 'Tokyo-Walk_rgb.mp4')

# 让脚本能 import 项目模块
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video


# ==================== 第 1 步：深度视频 ====================

def collect_inputs(input_path):
    """返回待处理视频文件列表：文件 -> 单个；文件夹 -> 其中所有 mp4"""
    if os.path.isdir(input_path):
        return sorted(glob.glob(os.path.join(input_path, '*.mp4')))
    return [input_path]


def depth_to_video(video_path, output_dir, args, model_configs, device):
    """彩色视频 → 深度可视化视频，返回 _vis.mp4 路径（失败返回 None）"""
    video_name = os.path.basename(video_path)
    print(f'\n===== [第 1 步] 深度估计: {video_path} =====')

    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'
    checkpoint_path = os.path.join(PROJECT, 'checkpoints', f'{checkpoint_name}_{args.encoder}.pth')
    if not os.path.exists(checkpoint_path):
        print(f'[错误] 权重文件不存在: {checkpoint_path}')
        return None

    # 加载模型
    model = VideoDepthAnything(**model_configs[args.encoder], metric=args.metric)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
    model = model.to(device).eval()

    # 读取视频帧
    print('读取视频帧...')
    frames, target_fps = read_video_frames(video_path, args.max_len, args.target_fps, args.max_res)
    print(f'共 {len(frames)} 帧, 帧率 {target_fps}')

    # 推理
    print('深度推理中（可能需要几分钟）...')
    depths, fps = model.infer_video_depth(
        frames, target_fps, input_size=args.input_size, device=device, fp32=args.fp32
    )

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(video_name)[0]
    processed_video_path = os.path.join(output_dir, base + '_src.mp4')
    depth_vis_path = os.path.join(output_dir, base + '_vis.mp4')
    print('保存结果视频...')
    save_video(frames, processed_video_path, fps=fps)
    save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)

    print(f'[第 1 步完成] {video_name}')
    print(f'  源视频: {processed_video_path}')
    print(f'  深度可视化: {depth_vis_path}')

    if args.save_npz:
        depth_npz_path = os.path.join(output_dir, base + '_depths.npz')
        np.savez_compressed(depth_npz_path, depths=depths)
        print(f'  深度数据: {depth_npz_path}')

    return depth_vis_path


# ==================== 第 2 步：雪花噪声视频 ====================

def read_depth_frames(video_path):
    """读深度视频所有帧的灰度亮度（近处亮远处暗），返回 (frames, fps, w, h)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'无法打开深度视频: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise RuntimeError(f'深度视频没有可读帧: {video_path}')
    return frames, fps, w, h


def depth_range(frames, low=2.0, high=98.0):
    """用百分位求深度亮度的全局有效范围"""
    sample = np.concatenate([f.ravel()[::37] for f in frames])
    return np.percentile(sample, low), np.percentile(sample, high)


def depth_video_to_snow(depth_video_path, args):
    """深度视频 → 雪花噪声视频（时间频率编码），返回输出路径"""
    print(f'\n===== [第 2 步] 雪花噪声: {depth_video_path} =====')
    output = os.path.splitext(depth_video_path)[0] + '_static.mp4'

    frames, src_fps, w, h = read_depth_frames(depth_video_path)
    print(f'  共 {len(frames)} 帧, {w}x{h}, 源帧率 {src_fps:.1f}')

    lo, hi = depth_range(frames)
    if hi - lo < 1:
        hi = lo + 1
    print(f'  深度亮度范围: {lo:.0f} ~ {hi:.0f}')

    np.random.seed(args.seed)
    per_src = max(1, round(args.snow_fps / src_fps))

    import imageio.v2 as iio
    writer = iio.get_writer(output, fps=args.snow_fps, macro_block_size=1,
                            codec='libx264', ffmpeg_params=['-crf', '18'])
    print(f'  生成雪花视频: {output}  ({args.snow_fps}fps, '
          f'far_th={args.far_th}, near_th={args.near_th}, max_rate={args.max_rate})')

    if args.color:
        noise_state = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    else:
        noise_state = np.random.randint(0, 256, (h, w), dtype=np.uint8)

    total_out = len(frames) * per_src
    done = 0
    for gray in frames:
        d = (gray.astype(np.float32) - lo) / (hi - lo)
        d = np.clip(d, 0.0, 1.0)
        # 分段饱和映射: 远=静止(0), 中=平滑陡升, 近=饱和(最大)
        p = np.zeros_like(d, dtype=np.float32)
        mid = (d > args.far_th) & (d < args.near_th)
        t = np.clip((d[mid] - args.far_th) / (args.near_th - args.far_th), 0.0, 1.0)
        p[mid] = t * t * (3.0 - 2.0 * t)
        p[d >= args.near_th] = 1.0
        p *= args.max_rate

        for _ in range(per_src):
            new_random = np.random.randint(0, 256, noise_state.shape, dtype=np.uint8)
            if args.color:
                update = np.random.random((h, w, 3)) < p[:, :, None]
            else:
                update = np.random.random((h, w)) < p
            noise_state = np.where(update, new_random, noise_state)
            writer.append_data(noise_state)
            done += 1
            if done % 60 == 0 or done == total_out:
                print(f'  进度 {done}/{total_out}', end='\r')
    writer.close()
    print(f'\n[第 2 步完成] 雪花视频: {output}')
    return output


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description='两步流水线: 彩色视频 → 深度视频 → 雪花噪声视频')
    parser.add_argument('--input_video', type=str, default=DEFAULT_VIDEO,
                        help=f'视频路径或文件夹，默认: {DEFAULT_VIDEO}')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='输出目录，默认视频所在文件夹')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'],
                        help='深度模型大小（vits 最快，vitl 精度最高）')
    # 第 1 步参数
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--max_len', type=int, default=-1, help='最大处理帧数，-1 无限制')
    parser.add_argument('--target_fps', type=int, default=-1, help='深度推理目标帧率，-1 保持原帧率')
    parser.add_argument('--metric', action='store_true', help='使用度量深度模型')
    parser.add_argument('--fp32', action='store_true', help='使用 float32 推理（默认 float16）')
    parser.add_argument('--grayscale', action='store_true', help='深度图不套彩色调色板')
    parser.add_argument('--save_npz', action='store_true', help='同时保存深度为 npz')
    # 第 2 步参数
    parser.add_argument('--no_snow', action='store_true', help='跳过雪花生成，只做深度视频')
    parser.add_argument('--far_th', type=float, default=0.25,
                        help='深度≤此值视为背景，雪花完全静止（默认 0.25）')
    parser.add_argument('--near_th', type=float, default=0.55,
                        help='深度≥此值直接饱和为最大更新（默认 0.55）')
    parser.add_argument('--max_rate', type=float, default=1.0,
                        help='近处最大更新概率（默认 1 = 每帧都闪烁）')
    parser.add_argument('--snow_fps', type=int, default=30, help='雪花视频帧率（默认 30）')
    parser.add_argument('--color', action='store_true', help='雪花使用彩色噪声')
    parser.add_argument('--seed', type=int, default=12345, help='随机种子')
    args = parser.parse_args()

    if args.far_th >= args.near_th:
        print(f'[错误] --far_th ({args.far_th}) 必须小于 --near_th ({args.near_th})')
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_list = collect_inputs(args.input_video)
    if not video_list:
        print(f'[错误] 未找到待处理视频: {args.input_video}')
        sys.exit(1)

    print(f'待处理视频 {len(video_list)} 个:')
    for v in video_list:
        print(f'  - {v}')

    ok_depth = 0
    ok_snow = 0
    for video_path in video_list:
        try:
            depth_vis = depth_to_video(video_path, args.output_dir, args, model_configs, device)
            if depth_vis is None:
                continue
            ok_depth += 1
            if not args.no_snow:
                depth_video_to_snow(depth_vis, args)
                ok_snow += 1
        except Exception as e:
            import traceback
            print(f'[失败] {os.path.basename(video_path)}: {e}')
            traceback.print_exc()

    print(f'\n全部完成: 深度 {ok_depth}/{len(video_list)}, 雪花 {ok_snow}/{len(video_list)}')
    print(f'输出目录: {args.output_dir}')


if __name__ == '__main__':
    main()
