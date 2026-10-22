# Guqq2 网络配置

## 已验证的连接设置

- SSH 别名：`Guqq2`。
- 地址：`114.214.215.42:22`；用户：`xmz`。
- 跳板机：`vlab`。
- 本地私钥路径：`~/.ssh/id_ed25519_codex`；私钥保留在本机。
- 反向动态 SOCKS 转发：`127.0.0.1:1080`，仅回环可见。
- Keepalive：间隔 30 秒，连续失败 3 次后断开。
- 当前 Host 配置 `ExitOnForwardFailure no`；专用代理启动命令显式覆盖为 `yes`。

## 每次需要联网时

先从普通工作连接检查是否已有健康代理。已有代理健康时复用，不重复启动。

本机 PowerShell 专用代理会话（持续保持运行）：

```powershell
ssh -N -T -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 Guqq2
```

另开本机 PowerShell 普通工作连接：

```powershell
ssh -o ClearAllForwardings=yes Guqq2
```

在服务器工作连接中检查：

```bash
ss -ltn 'sport = :1080'
curl --proxy socks5h://127.0.0.1:1080 --head --connect-timeout 10 --max-time 25 https://github.com/
```

服务器联网命令显式使用代理，例如：

```bash
ALL_PROXY=socks5h://127.0.0.1:1080 \
HTTP_PROXY=socks5h://127.0.0.1:1080 \
HTTPS_PROXY=socks5h://127.0.0.1:1080 \
git pull --ff-only
```

该 git 命令应在已确认的任务仓库目录中运行。本次本地目录没有 Git 元数据或已确认 remote，因此未执行远程仓库同步。

不将代理变量写入服务器全局或登录配置，以免专用连接断开后残留失效代理。Slurm 计算节点的网络路径需单独验证，不能直接使用登录节点的回环地址。

工作完成后，在自己创建的专用代理会话按 Ctrl+C，然后从禁用转发的工作连接确认 1080 监听释放。保留其他用户会话。

## 本次验证

- 非交互 SSH 认证成功，经 vlab 到达 Guqq2。
- 首次检查无 1080 监听；没有终止已有的用户普通 SSH 会话。
- 新建专用代理后监听为 `127.0.0.1:1080`，GitHub HTTPS 返回 `HTTP/2 200`。
- SSH Host 配置与 AGENTS.md 已使用 Guqq2，本次无需修改它们。
- 验证结束后已关闭本任务专用代理，复核 1080 无监听。
