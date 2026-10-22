# 上游代码

已通过真实 git clone 获取 [AkiraTOSEI/SMOACS](https://github.com/AkiraTOSEI/SMOACS)，本地目录为 `SMOACS/`。

- 固定版本：`24fe8467de87167818e75ccd7eec21ab673f8df5`。
- 上游许可：MIT；附带第三方代码以其各自许可证为准，见 clone 中的 LICENSE 和 THIRD_PARTY_NOTICES.md。
- clone 未修改上游源码；没有发现达到 10 MiB 的文件。
- 已登记为父仓库正式子模块；.gitmodules 记录 origin，gitlink 固定该 commit。

重建当前版本：

```powershell
git clone https://github.com/AkiraTOSEI/SMOACS.git third_party/SMOACS
git -C third_party/SMOACS checkout 24fe8467de87167818e75ccd7eec21ab673f8df5
```

新 checkout 使用 git submodule update --init --recursive 获取固定源码；依赖、权重、模板、日志与结果继续遵守生成物忽略规则。部署到服务器前检查子模块 commit，并按 AGENTS.md 同步。
