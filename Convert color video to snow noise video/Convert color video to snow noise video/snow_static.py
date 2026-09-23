# -*- coding: utf-8 -*-
"""
snow_static.py — 基于深度视频生成 "Lost in the Static" 风格雪花噪声视频
========================================================================
原理（时间频率编码）：
  屏幕上每个像素都是完全均匀的随机雪花亮度（单帧暂停看不出任何结构）。
  像素的"更新频率"由深度分段决定：
    - 深度 ≤ far_th   → 背景，雪花信号完全静止（永不更新）
    - 深度 ≥ near_th  → 前景，直接饱和为最大更新概率（每帧闪烁）
    - 中间深度        → 平滑陡峭过渡（从静止快速升到最大）
  播放时人眼通过运动视差感知深度：背景安静如画、近处躁动闪烁。

用法：
  D:\anaconda3\envs\vda\python.exe snow_static.py
  D:\anaconda3\envs\vda\python.exe snow_static.py --input <深度视频路径>
  D:\anaconda3\envs\vda\python.exe snow_static.py --far_th 0.2 --near_th 0.5
  D:\anaconda3\envs\vda\python.exe snow_static.py --color   # 彩色雪花
"""
import argparse
import os
import sys

# 保证中文输出在 Windows 控制台正常
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(os.path.dirname(ROOT), 'Videos Pending Processing')
DEFAULT_INPUT = os.path.join(VIDEO_DIR, 'Tokyo-Walk_rgb_vis.mp4')


def read_depth_frames(video_path):
    """读视频所有帧的灰度亮度（深度视频近处亮远处暗），返回 (frames, fps, w, h)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频: {video_path}')
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
        raise RuntimeError(f'视频没有可读帧: {video_path}')
    return frames, fps, w, h


def depth_range(frames, low=2.0, high=98.0):
    """用百分位求深度亮度的全局有效范围，避免极端值干扰"""
    sample = np.concatenate([f.ravel()[::37] for f in frames])
    return np.percentile(sample, low), np.percentile(sample, high)


def main():
    parser = argparse.ArgumentParser(description='生成 Lost in the Static 风格雪花噪声视频（时间频率编码）')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT,
                        help=f'深度视频路径，默认: {DEFAULT_INPUT}')
    parser.add_argument('--output', type=str, default='',
                        help='输出视频路径，默认与输入同目录 *_static.mp4')
    parser.add_argument('--far_th', type=float, default=0.25,
                        help='深度≤此值视为背景，雪花完全静止（默认 0.25）')
    parser.add_argument('--near_th', type=float, default=0.55,
                        help='深度≥此值直接饱和为最大更新（默认 0.55）')
    parser.add_argument('--max_rate', type=float, default=1.0,
                        help='近处最大更新概率（默认 1 = 每帧都闪烁，范围 0-1）')
    parser.add_argument('--fps', type=int, default=30,
                        help='输出帧率（默认 30，越高近处闪烁越流畅）')
    parser.add_argument('--color', action='store_true', help='使用彩色雪花噪声')
    parser.add_argument('--seed', type=int, default=12345, help='随机种子')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'[错误] 深度视频不存在: {args.input}')
        print('提示: 请先用 1.py 生成深度视频（如 Tokyo-Walk_rgb_vis.mp4）')
        sys.exit(1)

    if args.far_th >= args.near_th:
        print(f'[错误] --far_th ({args.far_th}) 必须小于 --near_th ({args.near_th})')
        sys.exit(1)

    output = args.output or os.path.splitext(args.input)[0] + '_static.mp4'

    print(f'读取深度视频: {args.input}')
    frames, src_fps, w, h = read_depth_frames(args.input)
    print(f'  共 {len(frames)} 帧, {w}x{h}, 源帧率 {src_fps:.1f}')

    print('计算深度全局范围...')
    lo, hi = depth_range(frames)
    if hi - lo < 1:
        hi = lo + 1
    print(f'  亮度范围: {lo:.0f} ~ {hi:.0f}')

    np.random.seed(args.seed)
    out_fps = args.fps
    per_src = max(1, round(out_fps / src_fps))  # 每输入帧生成几个输出帧

    import imageio.v2 as iio
    writer = iio.get_writer(output, fps=out_fps, macro_block_size=1,
                            codec='libx264', ffmpeg_params=['-crf', '18'])
    print(f'生成雪花噪声视频: {output}  ({out_fps}fps, '
          f'far_th={args.far_th}, near_th={args.near_th}, max_rate={args.max_rate})')

    # 初始化噪声状态：所有像素均匀随机（单帧看不出任何结构）
    if args.color:
        noise_state = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    else:
        noise_state = np.random.randint(0, 256, (h, w), dtype=np.uint8)

    total_out = len(frames) * per_src
    done = 0
    for gray in frames:
        # 深度归一化: 近处=1, 远处=0
        d = (gray.astype(np.float32) - lo) / (hi - lo)
        d = np.clip(d, 0.0, 1.0)

        # 分段饱和映射: 远=静止(0), 中=平滑陡升, 近=饱和(最大)
        p = np.zeros_like(d, dtype=np.float32)
        mid = (d > args.far_th) & (d < args.near_th)
        t = np.clip((d[mid] - args.far_th) / (args.near_th - args.far_th), 0.0, 1.0)
        p[mid] = t * t * (3.0 - 2.0 * t)                    # smoothstep
        p[d >= args.near_th] = 1.0                          # 近处直接饱和
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
    print(f'\n[完成] 输出: {output}')

    print('小提示: 播放时背景雪花钉住不动，近处物体快速闪烁；单帧暂停看不出结构属正常。'
          '若感觉不适请停止观看。')


if __name__ == '__main__':
    main()
