# 服务器连接记录

- 2026-10-22 11:52:26 +08:00：计划检查本地 SSH 配置及进程；连接 Guqq2 进行公钥认证、1080 回环监听、代理 HTTPS 健康检查。属于 AGENTS.md 允许的网络诊断例外；在仓库同步门禁前不操作代码、实验或作业。若需要，将建立一个专用代理持有会话，并在完成后关闭本任务创建的会话。

- 2026-10-22 11:53:10：Guqq2 BatchMode 登录成功；1080 无监听，curl 返回 Connection refused。现有本地 ssh Guqq2 为用户普通会话，保持不动。下一次连接建立独立 -N -T 专用代理，并用单独的禁转发连接复测 HTTPS。

- 2026-10-22 11:53:34：专用代理会话已启动（工具会话 59813）。下一连接仅执行 1080 监听及 GitHub HTTPS 端到端复核。

- 2026-10-22 11:54:29 +08:00：专用代理端到端验证成功：127.0.0.1:1080 LISTEN，GitHub HTTPS HTTP/2 200。准备关闭仅本任务创建的工具会话 59813；下一次连接仅确认回环监听释放。

- 2026-10-22 11:55:16 +08:00：清理检查通过：专用代理会话已结束，最终 ss 仅输出表头，1080 监听已释放。

- 2026-10-22 12:20:36 +08:00：应用户明确请求，连接 Guqq2 只读查询 nvidia-smi、Slurm 节点与作业状态，确认 RTX 5090 数量和占用；不操作任务仓库、不提交作业、不启动代理。

- 首次 GPU 查询未执行成功：vlab 跳板机主动关闭连接，未取得硬件或 Slurm 信息。下一连接重试相同只读检查，设置连接超时与 keepalive。

- 2026-10-22 12:23:59 +08:00：Guqq2（node42）检测到 1 张 NVIDIA GeForce RTX 5090；显存总量 32607 MiB、已用 184 MiB、GPU 利用率 0%。Slurm 仅列出 node42，Gres=gpu:1，分区 compute，State=IDLE，squeue 无作业。 首次连接被跳板机断开，第二次查询成功，退出码 0。

- 2026-10-30T16:25:26: Pre-submission synchronization authorization: local tested source will be committed/pushed and copied as verified Git bundle, remote mandatory pull then Slurm GPU sampling/ALIGNN/MatterSim single points and dependent CPU audit/archive only. No login-node model compute or internet required.

- 2026-10-30T16:29:45: Authorize source-bundle update for phase analysis only, mandatory verified pull, submit new CPU phase job depending afterok506. Existing native GPU experiment505 remains unchanged.

- 2026-10-30T16:40:45: Authorize reporting-only source synchronization and mandatory verified pull, CPU Slurm rerender/phase summary with existing505rawdata; preserve complete.json and alltrajectory hashes, no GPU compute.

- 2026-10-30T16:47:30: Authorize tested manifest-only source synchronization, mandatory verified pull and one CPU phase/audit job on existing raw results. No remote source editing, GPU load, or model recomputation.

- 2026-10-30T16:52:08: Remote work complete: successful GPU505 and CPU506-510; final verified archive downloaded. All compute used Slurm. Ordinary SSH/SCP processes exited; no task-owned dedicated proxy holder was started or left running. No further connection needed for model work.

- 2026-10-31T15:57:25: Authorize source sync after tested commit, verified mandatory bundle pull, SlurmCPUdatasetselection followed byGPUfrozenfeatureextraction. Source remains local-only edited; results new alignn-reliability-20261031. No network/proxy/download needed.

- 2026-10-31T16:03:46: Authorize report-only source sync/verified pull and SlurmCPUtrain+audit+archive afterok512. Running feature source unchanged; no additional dataset download, DFT or predictor retraining.

- 2026-10-31T16:17:52: Authorize diagnostic-only verified source sync/pull and short SlurmGPU current-native versus archived/batched check on7posthocIDs. No classifier/predictor retraining or DFT. Outputs same task result directory, no overwrite of trained heads.
