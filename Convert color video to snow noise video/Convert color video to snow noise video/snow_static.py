# -*- coding: utf-8 -*-
"""
snow_static.py — 基于深度视频生成 "Lost in the Static" 风格雪花噪声视频
========================================================================
三种模式（--mode 1 / 2 / 3）：

模式 1（--mode 1，最早的频率版）：
  所有像素均匀随机噪声，深度只控制"更新频率"：
  远(≤ far_th)完全静止、中(smoothstep 陡升)、近(≥ near_th)每帧闪烁。
  播放时轮廓浮现，单帧看不出结构。

模式 2（--mode 2，噪声场平移方向渐变版，默认）：
  预生成一张比屏幕大得多的均匀随机噪声场，每个像素从场中采样窗口；
  背景采样窗口固定（雪花完全静止），近处采样窗口沿该深度对应的方向
  匀速平移（一片近景向下、相邻近景向上……），方向随深度渐变。
  单帧暂停所有区域是完全相同的雪花（无空间相关、无模糊）。

模式 3（--mode 3，近处固定向上版）：
  同样基于噪声场平移；不同点：深度 ≥ near_th 的近处雪花**始终向上移动**，
  far_th~near_th 的过渡区方向随深度渐变（从"下"逐渐转到"上"，平滑衔接）。

用法：
  python snow_static.py --mode 2
  python snow_static.py --input <深度视频路径> --mode 3
  python snow_static.py --far_th 0.25 --near_th 0.55 --directions 8 --speed 2
  python snow_static.py --color          # 彩色雪花
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
DEFAULT_INPUT = os.path.join(VIDEO_DIR, 'test_vis.mp4')

# 8 邻域方向（vx, vy）= 雪花内容在屏幕上的移动方向：
# 0..7 依次为 上、右上、右、右下、下、左下、左、左上
# 采样偏移 = -方向（内容沿 (vx,vy) 移动 → 采样窗口反向扫描）
DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
VX = np.array([d[0] for d in DIRS], dtype=np.float32)
VY = np.array([d[1] for d in DIRS], dtype=np.float32)


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


def update_probability(d, far_th, near_th, max_rate=1.0):
    """模式 1：深度 → 更新概率。远=0(静止) → 中=smoothstep 陡升 → 近=饱和"""
    p = np.zeros_like(d, dtype=np.float32)
    mid = (d > far_th) & (d < near_th)
    t = np.clip((d[mid] - far_th) / (near_th - far_th), 0.0, 1.0)
    p[mid] = t * t * (3.0 - 2.0 * t)
    p[d >= near_th] = 1.0
    return p * max_rate


def speed_map(d, far_th, near_th, speed):
    """模式 2/3：深度 → 平移速度。背景=0(静止) → 中=smoothstep 陡升 → 近=饱和(全速)"""
    s = np.zeros_like(d, dtype=np.float32)
    mid = (d > far_th) & (d < near_th)
    t = np.clip((d[mid] - far_th) / (near_th - far_th), 0.0, 1.0)
    s[mid] = t * t * (3.0 - 2.0 * t)
    s[d >= near_th] = 1.0
    return s * speed


def direction_map(d, far_th, near_th, directions, mode):
    """模式 2/3：深度 → 方向编号（= 雪花内容移动方向）

    模式 2：far_th 以上随深度渐变 0..directions-1（方向随深度线性展开）
    模式 3：far_th~near_th 过渡区从"下"(idx=4) 渐变到"上"(idx=0)（平滑衔接），
            ≥ near_th 的近处固定"上"（方向 0）
    """
    if mode == 3:
        idx = np.zeros_like(d, dtype=np.int32)
        mid = (d > far_th) & (d < near_th)
        tm = np.clip((d[mid] - far_th) / (near_th - far_th), 0.0, 1.0)
        idx[mid] = ((1.0 - tm) * (directions // 2)).astype(np.int32) % directions
        idx[d >= near_th] = 0          # 近处固定向上
        return idx
    # 模式 2
    t = np.clip((d - far_th) / max(1e-6, 1.0 - far_th), 0.0, 1.0)
    return (t * directions).astype(np.int32) % directions


def main():
    parser = argparse.ArgumentParser(description='生成 Lost in the Static 风格雪花噪声视频（3 种模式）')
    parser.add_argument('--mode', type=int, default=3, choices=[1, 2, 3],
                        help='雪花模式: 1=频率版(轮廓可见)  2=方向渐变版(默认)  3=近处固定向上')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT,
                        help=f'深度视频路径，默认: {DEFAULT_INPUT}')
    parser.add_argument('--output', type=str, default='',
                        help='输出视频路径，默认与输入同目录 *_static_m<mode>.mp4')
    parser.add_argument('--far_th', type=float, default=0.25,
                        help='深度≤此值视为背景，雪花完全静止（默认 0.25）')
    parser.add_argument('--near_th', type=float, default=0.55,
                        help='深度≥此值全速（模式1饱和闪烁；模式2/3平移全速）（默认 0.55）')
    parser.add_argument('--speed', type=float, default=2.0,
                        help='模式2/3：近处雪花平移速度，像素/帧（默认 2，越大方向感越强）')
    parser.add_argument('--directions', type=int, default=8, choices=[4, 8, 16],
                        help='模式2/3：方向量化档数（默认 8）')
    parser.add_argument('--fps', type=int, default=30,
                        help='输出帧率（默认 30）')
    parser.add_argument('--color', action='store_true', help='使用彩色雪花噪声')
    parser.add_argument('--seed', type=int, default=12345, help='随机种子')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'[错误] 深度视频不存在: {args.input}')
        print('提示: 请先用 run.py 或 1.py 生成深度视频（如 Tokyo-Walk_rgb_vis.mp4）')
        sys.exit(1)

    if args.far_th >= args.near_th:
        print(f'[错误] --far_th ({args.far_th}) 必须小于 --near_th ({args.near_th})')
        sys.exit(1)

    if not args.output:
        base, _ = os.path.splitext(args.input)
        args.output = f'{base}_static_m{args.mode}.mp4'

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
    per_src = max(1, round(out_fps / src_fps))   # 每输入帧生成几个输出帧
    total_out = len(frames) * per_src

    import imageio.v2 as iio
    writer = iio.get_writer(args.output, fps=out_fps, macro_block_size=1,
                            codec='libx264', ffmpeg_params=['-crf', '18'])
    print(f'生成雪花视频: {args.output}  ({out_fps}fps, mode={args.mode}, '
          f'far_th={args.far_th}, near_th={args.near_th}, '
          f'directions={args.directions}, speed={args.speed})')

    shape = (h, w, 3) if args.color else (h, w)

    if args.mode == 1:
        # ===== 模式 1：时间频率版（远静止、近闪烁）=====
        noise_state = np.random.randint(0, 256, shape, dtype=np.uint8)
        done = 0
        for gray in frames:
            d = (gray.astype(np.float32) - lo) / (hi - lo)
            d = np.clip(d, 0.0, 1.0)
            p = update_probability(d, args.far_th, args.near_th)
            for _ in range(per_src):
                new_random = np.random.randint(0, 256, shape, dtype=np.uint8)
                if args.color:
                    update = np.random.random((h, w, 3)) < p[:, :, None]
                else:
                    update = np.random.random((h, w)) < p
                noise_state = np.where(update, new_random, noise_state)
                writer.append_data(noise_state)
                done += 1
                if done % 60 == 0 or done == total_out:
                    print(f'  进度 {done}/{total_out}', end='\r')
    else:
        # ===== 模式 2/3：大噪声场平移方向版 =====
        margin = int(total_out * args.speed) + 16
        field_h, field_w = h + 2 * margin, w + 2 * margin
        if args.color:
            field = np.random.randint(0, 256, (field_h, field_w, 3), dtype=np.uint8)
        else:
            field = np.random.randint(0, 256, (field_h, field_w), dtype=np.uint8)
        print(f'  噪声场尺寸: {field_w}x{field_h} (余量 {margin}px)')

        yy, xx = np.mgrid[0:h, 0:w]
        t_out = 0
        done = 0
        for gray in frames:
            d = (gray.astype(np.float32) - lo) / (hi - lo)
            d = np.clip(d, 0.0, 1.0)
            dir_idx = direction_map(d, args.far_th, args.near_th, args.directions, args.mode)
            vx = VX[dir_idx]
            vy = VY[dir_idx]
            sp = speed_map(d, args.far_th, args.near_th, args.speed)
            for _ in range(per_src):
                # 采样窗口位置 = 初始 - t * 内容移动方向 * 速度（按 t 直接计算，无累积误差）
                oy = (margin - t_out * vy * sp).astype(np.int32)
                ox = (margin - t_out * vx * sp).astype(np.int32)
                writer.append_data(field[yy + oy, xx + ox])
                t_out += 1
                done += 1
                if done % 60 == 0 or done == total_out:
                    print(f'  进度 {done}/{total_out}', end='\r')
    writer.close()
    print(f'\n[完成] 输出: {args.output}')

    print('小提示: 播放时背景雪花钉住不动，模式2/3 近处雪花按方向整体移动；'
          '单帧暂停所有区域是完全相同的雪花。若感觉不适请停止观看。')


if __name__ == '__main__':
    main()
