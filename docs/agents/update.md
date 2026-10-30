
- 2026-10-22 11:54:29 +08:00：验证 Guqq2 免密与反向 SOCKS 网络成功，新增 network.md 使用说明及任务记录；未修改 SSH 配置、服务器源码、用户普通 SSH 会话或提交计算任务。

- 2026-10-22 11:55:16 +08:00：清理检查通过：专用代理会话已结束，最终 ss 仅输出表头，1080 监听已释放。

- 2026-10-22 12:23:59 +08:00：Guqq2（node42）检测到 1 张 NVIDIA GeForce RTX 5090；显存总量 32607 MiB、已用 184 MiB、GPU 利用率 0%。Slurm 仅列出 node42，Gres=gpu:1，分区 compute，State=IDLE，squeue 无作业。 首次连接被跳板机断开，第二次查询成功，退出码 0。

- 2026-10-30T16:25:26: Implemented conditional_trajectory module: native joint bandgap CFG2/CFG0 paired sampling, RNG-neutral stride5 capture, actual terminal output, separate ALIGNN/MatterSim posthoc inference, support/mask/geometry coverage, paired and fixed-cohort target MAE, raw audit and plot exports. Local tests pass; next synchronize and submit Slurm GPU plus dependent CPU audit.
