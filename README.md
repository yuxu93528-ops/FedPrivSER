# FedPriv-SER
FedPriv-SER 是一个面向 RAVDESS Speech Audio 的可复现实验项目，用于完成“语音情感识别 + 联邦学习 + 隐私扰动 + 对抗去身份化 + 隐私攻击评估”的完整实验闭环。

本项目是隐私保护的大作业的相关的实验结果和代码

## 1. 项目简介

- 数据集：RAVDESS Speech Audio
- 每个 actor 视为一个 federated client，共 24 个客户端
- 特征：40 维 MFCC 及其 delta / delta2 的均值和标准差，共 240 维
- 模型：轻量级 MLP
- 方法：
  - Centralized
  - Local-only
  - FedAvg
  - FedAvg + DP-inspired client update perturbation
  - FedAvg + GRL
  - FedAvg + GRL + DP（FedPriv-SER）

## 2. 数据集放置路径

项目默认读取：

`E:\2\Audio_Speech_Actors_01-24`

请不要复制数据集到项目目录，路径统一由 [config.py](/E:/2/FedPriv-SER/src/config.py) 管理。

## 3. 环境安装

建议 Python 3.10 或 3.11：

```bash
pip install -r requirements.txt
```

## 4. CPU / GPU 说明

- 默认自动检测 `torch.cuda.is_available()`
- 检测到 CUDA 时自动使用 GPU
- 否则自动回退到 CPU
- 不依赖 Linux、Docker、Hugging Face、wav2vec2、HuBERT、Whisper

若使用 RTX 3080 10GB，可用以下命令确认 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 5. 运行步骤

数据检查：

```bash
python -m src.check_data
```

特征提取：

```bash
python -m src.extract_features
```

单个实验：

```bash
python -m src.train_centralized
python -m src.train_local_only
python -m src.train_fedavg
python -m src.train_fedavg_dp
python -m src.train_fedavg_grl
python -m src.train_fedavg_grl_dp
```

扩展实验：

```bash
python -m src.exp_fedavg_convergence
python -m src.exp_grl_alpha_sweep
python -m src.exp_grl_schedule
python -m src.exp_dp_sweep
python -m src.exp_dual_grl
python -m src.update_level_attacks
python -m src.exp_multi_seed
python -m src.summarize_extended_results
```

一键运行全部实验：

```bash
python run_all.py
```

## 6. 输出结果

- 特征文件：
  - `features/ravdess_metadata.csv`
  - `features/ravdess_mfcc_features.csv`
- 表格：
  - `results/tables/main_emotion_results.csv`
  - `results/tables/privacy_attack_results.csv`
  - `results/tables/privacy_utility_tradeoff.csv`
  - `results/tables/client_level_results.csv`
  - `results/tables/data_split_summary.csv`
- 图像：
  - `results/figures/emotion_results_bar.png`
  - `results/figures/privacy_attack_bar.png`
  - `results/figures/privacy_utility_curve.png`
  - `results/figures/confusion_matrix_fedpriv_ser.png`
  - `results/figures/fedavg_training_curve.png`
  - `results/figures/fedavg_convergence_curve.png`
  - `results/figures/grl_alpha_privacy_utility.png`
  - `results/figures/update_level_attack_bar.png`
  - `results/figures/dp_sweep_privacy_utility.png`
  - `results/figures/dual_grl_privacy_utility.png`
- 日志与 checkpoint：
  - `results/logs/`
  - `results/checkpoints/`

扩展表格：

- `results/tables/fedavg_convergence_results.csv`
- `results/tables/grl_alpha_sweep.csv`
- `results/tables/grl_schedule_results.csv`
- `results/tables/update_level_attack_results.csv`
- `results/tables/dp_sweep_results.csv`
- `results/tables/dual_grl_results.csv`
- `results/tables/multi_seed_results.csv`
- `results/tables/multi_seed_summary.csv`
- `results/tables/final_selected_results.csv`

## 7. Baseline 含义

- `Centralized`：所有客户端训练集汇总后统一训练
- `Local-only`：每个客户端独立训练并在本地测试集评估后求均值
- `FedAvg`：标准联邦平均
- `FedAvg + DP`：客户端更新先裁剪再加高斯噪声
- `FedAvg + GRL`：在共享表征上加入 speaker adversarial branch
- `FedPriv-SER`：GRL 与 client-level update perturbation 同时使用

## 8. 复现说明

- 所有随机种子固定为 42
- 所有路径集中在 [config.py](/E:/2/FedPriv-SER/src/config.py)
- 所有实验共享同一份数据划分
- `StandardScaler` 仅在训练集上拟合
- 若特征文件已存在，默认不重复提取

## 9. FAST_DEBUG

在 [config.py](/E:/2/FedPriv-SER/src/config.py) 中可设置：

- `FAST_DEBUG = True`

用于降低样本量、epoch 和联邦轮数，方便快速调试。

## 10. 扩展实验说明

- `FedAvg convergence`：检查 `rounds / local_epochs / learning_rate` 是否导致联邦训练收敛不足。
- `GRL alpha sweep`：扫描不同对抗强度，寻找更合适的 representation-level 隐私效用平衡点。
- `GRL schedule`：比较固定 GRL 与动态调度 GRL 的稳定性。
- `Update-level attack`：以客户端上传更新为输入，评估 update leakage。
- `DP sweep`：同时考察 `sigma` 与 `clip_norm` 对表示泄漏和更新泄漏的影响。
- `Dual-GRL`：同时对 speaker 与 gender 建立对抗分支。
- `Multi-seed`：用多个随机种子估计实验稳定性。

## 11. 两类隐私泄漏通道

- `representation attack`：输入中间表示 `z`，评估表征空间是否仍保留 speaker / gender 线索。
- `update-level attack`：输入客户端上传更新的统计特征，评估联邦通信过程是否泄漏身份信息。

经验上应分开分析：

- `GRL` 主要针对 `representation leakage`
- `DP-inspired update perturbation` 主要针对 `update leakage`

如果某个 DP 设置没有降低 `representation attack`，不能直接判定它完全失败，还需要结合 `update-level attack` 一起看。

## 12. 当前局限

- 本项目中的 DP 部分是 `client-level clipping + Gaussian perturbation`，未实现严格 privacy accountant，因此不应宣称为严格 epsilon-DP
- 隐私攻击默认基于测试集表征 `z` 训练 Logistic Regression 攻击器，适合课程论文复现实验，但仍可继续扩展为更严格的 attack protocol
- 最终结果必须以实际运行结果为准，不应伪造；若观察结果与预期不同，应在论文中解释原因
