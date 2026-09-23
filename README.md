# Convert Color Video to Snow Noise Video

将普通彩色视频转换成 **"Lost in the Static" 风格**的雪花噪声视频：播放时画面看似纯雪花，但人眼通过视觉残留，能感知到隐藏在噪声中的深度结构——远处的雪花**静止不动**，近处的雪花**快速闪烁**，场景随运动浮现。

> 灵感与机制参考：[Lost in the Static](https://silverspaceship.com/static/)（Silverspaceship Software）。这不是随机点立体图，不需要斗鸡眼。

## 效果原理

单帧画面中，每个像素都是**完全均匀的随机雪花亮度**——暂停任意一帧，看不出任何结构。但每个像素的**更新频率**由深度决定：

| 深度 | 雪花行为 |
|---|---|
| 距离远（≤ `far_th`） | **完全静止**，作为背景 |
| 中间深度 | smoothstep 陡峭过渡（从静止快速升到活跃） |
| 距离近（≥ `near_th`） | **直接饱和为最大更新**，每帧快速闪烁 |

播放时人眼对快速闪烁做时间积分（视觉残留），背景安静如画、近处躁动闪烁，深度层次就从噪声中"浮"出来。

## 两步流水线

```
彩色视频 ──► 深度视频 ──► 雪花噪声视频
            (第 1 步)      (第 2 步)
```

- **第 1 步**：使用 [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)（CVPR 2025 Highlight）估计每帧深度，生成深度可视化视频
- **第 2 步**：基于深度视频，用时间频率编码生成雪花噪声视频

## 目录结构

```
Convert color video to snow noise video/
├── run.py                            # 两步流水线主脚本（推荐入口）
├── Convert color video to snow noise video/
│   └── snow_static.py                 # 雪花噪声生成脚本（单独使用）
├── Video-Depth-Anything-main/         # 深度估计模型（第三方，Apache-2.0）
│   ├── checkpoints/                   # 预训练权重（需自行下载，见下文）
│   └── ...
└── Videos Pending Processing/         # 待处理视频 / 输出目录
```

## 环境要求

- Windows / Linux / macOS
- Python 3.10 / 3.11（`torch==2.1.1` 不支持 Python 3.12）
- NVIDIA GPU（可选，有 GPU 大幅加速；无 GPU 用 CPU 也能跑，但很慢）

### 安装

```bash
# 建议使用 conda 创建环境
conda create -n vda python=3.11 -y
conda activate vda

# 安装依赖（Windows 可跳过 xformers，代码会自动回退）
pip install -r Video-Depth-Anything-main/requirements.txt
```

> Windows 上 `xformers==0.0.23` 没有预编译包，安装时会失败。**无需担心**：项目代码中 xformers 的导入都在 `try/except` 内，缺失时自动回退到普通注意力实现，仅提示 `xFormers not available`，不影响结果。

### 下载预训练权重

删除的 `checkpoints` 目录需要手动下载权重（仓库不携带大文件）：

| 模型 | 下载地址 |
|---|---|
| Video-Depth-Anything-Large（相对深度） | [huggingface.co](https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth) |
| Metric-Video-Depth-Anything-Large（度量深度） | [huggingface.co](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth) |

（vits / vitb 版本见 [Video-Depth-Anything 官方页面](https://github.com/DepthAnything/Video-Depth-Anything#pre-trained-models)）

下载后放到：

```
Video-Depth-Anything-main/checkpoints/
├── video_depth_anything_vitl.pth
└── metric_video_depth_anything_vitl.pth
```

## 使用方法

**第 1 步**：激活环境（Windows / macOS / Linux 通用）：

```bash
conda activate vda
```

> 不想用 conda 的话，使用你自己创建的任何虚拟环境（`venv` 等）也可以，只要 Python 3.10/3.11 且依赖已安装。以下命令都在已激活的环境里执行。

**第 2 步**：把视频放进 `Videos Pending Processing/` 文件夹，然后运行：

```bash
# 一步完成：深度视频 + 雪花视频（vits 最快）
python run.py --encoder vits

# 追求精度（较慢）
python run.py --encoder vitl

# 处理指定视频或整个文件夹
python run.py --input_video "path/to/video.mp4" --encoder vits

# 只做深度，跳过雪花
python run.py --encoder vits --no_snow
```

### run.py 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input_video` | `Videos Pending Processing/Tokyo-Walk_rgb.mp4` | 输入视频或文件夹（批量处理其中所有 mp4） |
| `--output_dir` | 视频所在文件夹 | 输出目录 |
| `--encoder` | `vitl` | 深度模型：`vits`（快）/ `vitb` / `vitl`（准） |
| `--metric` | 关 | 使用度量深度模型（需对应权重） |
| `--target_fps` | -1 | 深度推理目标帧率（-1 保持原帧率，降低可加速） |
| `--max_res` | 1280 | 深度推理最大分辨率（降低可加速） |
| `--no_snow` | 关 | 跳过雪花生成 |
| `--far_th` | 0.25 | 深度≤此值视为背景，雪花完全静止 |
| `--near_th` | 0.55 | 深度≥此值直接饱和为最大更新 |
| `--max_rate` | 1.0 | 近处最大更新概率 |
| `--snow_fps` | 30 | 雪花视频帧率（越高闪烁越流畅） |
| `--color` | 关 | 彩色雪花模式 |
| `--seed` | 12345 | 随机种子（固定可复现） |

### 输出文件

在 `Videos Pending Processing/` 下生成：

```
Tokyo-Walk_rgb.mp4            # 原始输入
Tokyo-Walk_rgb_src.mp4        # 重编码后的源视频（与深度逐帧对应）
Tokyo-Walk_rgb_vis.mp4        # 第 1 步：深度可视化
Tokyo-Walk_rgb_vis_static.mp4 # 第 2 步：雪花噪声视频 ★
```

## 观看建议

- 播放时让眼睛在画面上扫视，不要死盯一个点，深度结构会浮现
- 近处物体闪烁明显、远处静止，运动视差是关键
- 如果感到头晕、恶心或眼睛不适，请立即停止观看（这是视觉残留的正常副作用，并非所有人都适合）

## 验证方式（可选）

对雪花视频做"帧间差分析"可验证效果：单帧统计应与纯随机噪声一致（mean≈127.5, std≈73.9），而连续 60 帧的绝对帧差图应呈现深度结构（背景帧差≈0、近处帧差大）。

## 许可证

- 本项目（`run.py`、`snow_static.py`、README 等）：[MIT License](LICENSE.md)
- 内嵌的 [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) 组件：**Apache License 2.0**（Copyright (c) 2025 Bytedance Ltd.），详见 `Video-Depth-Anything-main/LICENSE`

## 致谢

- [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) — 视频深度估计模型（CVPR 2025 Highlight）
- [Lost in the Static](https://silverspaceship.com/static/) — 视觉残留/雪花噪声创意启发
