# WSL训练运行与故障隔离

2026-10-07故障记录确认：旧planning.read_rows逐行读取NPZ成员，切片保留每次解压的完整数组，导致约21.6GiB RSS及swap耗尽。WSL init.scope的默认OOMPolicy=stop放大为实例退出。不是GPU显存不足或CPU性能不足；不能据此把全部历史Node崩溃都归为同一故障。

加载器现每个成员仅读取一次，行视图共享backing array；加载前检查ZIP解压总大小，默认不超过1GiB。回归测试统计实际成员访问次数，并验证数据和共享底层数组。2026-10-08完整11462行加载约1.08秒、峰值292MiB，包含CPU PyTorch导入，4GiB地址空间硬限制下成功。

CPU环境`.venv`保留，GPU环境`.venv-cuda`独立。训练入口支持`--device auto|cpu|cuda`，auto可回退CPU，明确指定cuda时不可用会报错，不能把CPU运行冒称为GPU。规则搜索与单局模拟主要在CPU，神经网络训练可迁移GPU。

## 独立限资源训练

```bash
scripts/train-planning-wsl.sh --epochs 12
# 从上一完整epoch恢复；--epochs为总目标轮数
scripts/train-planning-wsl.sh --epochs 12 --continue-training
systemctl --user list-units 'zimortal-planning-*' --all
ls -t logs/zimortal-planning-*.log
```

脚本创建临时systemd用户服务，MemoryMax=8GiB、MemorySwapMax=1GiB、CPUQuota=600%、Nice=10，OMP及OpenBLAS各2线程。默认使用独立GPU环境、显式cuda，保存到planning-cuda-resnet.pt及planning-cuda-pilot.json，与CPU试验权重分开。脚本通过flock拒绝重叠训练服务。服务独立于VS Code终端，日志写入项目logs；WSL整体退出仍会终止服务。没有改系统级OOM配置、WSL保活或驱动；用户systemd已验证可用。

模型和`.recovery.pt`均通过临时文件+原子替换保存。恢复包包含最后完整epoch的参数、优化器、Python/Torch/CUDA随机状态、最佳验证分数与日志、数据文件及父模型校验。数据或父模型改变拒绝恢复。CPU回归测试验证分两段训练与连续训练模型逐参数完全一致；不声称跨设备或跨PyTorch版本位级一致。中断会重做尚未保存的epoch。

CPU加载器默认1GiB解压预算是单文件保护，不代替整体训练cgroup上限；以后扩大数据需根据实测峰值评估流式加载，不能一味增加WSL分配内存。

CUDA安装依据：[PyTorch 2.7支持Blackwell及CUDA12.8](https://pytorch.org/blog/pytorch-2-7/)。本次最新cu128包下载依赖pypi.nvidia.com超时，转为官方2.7.1+cu128构建；实际GPU验证和训练结果补充到第七轮报告。

目标差距：运行稳定性不等于棋力提升，仍需完成现金收益、belief校准及搜索标签覆盖的独立评估。

完整CPU对照实测：同一数据12epoch，耗时15.56秒，峰值RSS 435.2MiB（不含其他进程）。训练服务4GiB上限下成功；不是历史OOM时的21.6GiB。报告：planning-cpu-benchmark.json。

独立GPU环境安装为PyTorch2.7.1+cu128，5080计算能力12.0。128状态批次已实际完成CUDA前向和反向，有限loss，峰值Torch张量显存约30MiB；不是仅检查nvidia-smi。CPU与CUDA环境分别199个测试通过。官方包直连下载成功，未持久改变代理配置。

GPU完整12epoch训练：16.52秒，进程峰值RSS 1.95GiB，Torch张量峰值显存 39.1MiB。8GiB服务上限下成功，init.scope没有新增OOM。显存数字不含驱动上下文和其他Windows应用。CPU/GPU环境PyTorch版本不同，这是一轮端到端观测，不是严格控制的硬件性能结论；当前小模型没有明显GPU加速，rollout和solver仍是CPU任务。

首次GPU完整验证发现soft target与指标输出设备混用，修复后重新完成12epoch；增加真实CUDA指标对照测试。最终CPU环境199通过、1个CUDA测试跳过，GPU可访问环境200通过。

GPU恢复实测：复制正式12epoch模型及恢复包到独立路径，加载后日志start_epoch=12，并成功继续第13轮；原12epoch权重及评估保持不变。恢复副本仅验证运行机制，不作为另一个棋力迭代计数。
