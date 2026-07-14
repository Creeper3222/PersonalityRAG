# PersonalityRAG Linux / Docker v0.1.0

PersonalityRAG 是独立运行的人格记忆 RAG 服务。本分支提供 Linux/Docker 发行版，与 Windows v0.1.0 保持核心能力、数据库、API、WebUI、召回和配置包兼容，并提供 amd64/arm64 多架构镜像。

- Windows 版与项目主页：[Creeper3222/PersonalityRAG](https://github.com/Creeper3222/PersonalityRAG)
- AstrBot 适配器：[astrbot_plugin_personality_rag_adapter](https://github.com/Creeper3222/astrbot_plugin_personality_rag_adapter)
- Docker Hub：`138763327/personalityrag-linux`

项目采用 AGPL-3.0-only。完整版权与第三方声明见 GitHub 仓库。

## 快速启动

```bash
cp .env.example .env
docker compose up -d
```

默认地址：

- WebUI：`http://127.0.0.1:8765/`
- AstrBot / 记忆库接入：`http://127.0.0.1:8766/`

正式 Compose 默认拉取：

```text
138763327/personalityrag-linux:latest
```

首次启动会在挂载的状态目录生成配置、密钥和默认记忆库。请从容器日志读取首次 API Key，并妥善备份状态目录。

## Docker Engine Socket 与版本切换

官方 Compose 将宿主机的 `/var/run/docker.sock` 挂载到容器，用于基础设置页中的版本检查、同版本重装、更新和回退。Docker Socket 等同于授予容器管理宿主 Docker Engine 的高权限；只应运行官方镜像，并限制 WebUI 的网络访问和登录凭据。

版本切换只识别 GitHub Release 中精确命名的 Linux 资产：

```text
PersonalityRAG-linux-vX.Y.Z.zip
```

Linux ZIP 只用于校验版本、源码提交和多架构镜像 digest。实际切换会拉取清单中当前 CPU 架构的不可变 Docker 镜像，并使用一次性的同镜像助手安全重建当前容器。`/app/state`、配置、密钥、数据库、FAISS 索引、备份和用户文件不会进入替换范围。

更新成功后会保留相同端口、环境变量、挂载、网络、Compose 标签和重启策略；新容器健康检查失败时会恢复旧容器和旧镜像。没有挂载 Docker Socket 时仍可检查版本，但切换按钮会被禁用。

## 本地开发镜像

```bash
cp .env.example .env
docker compose -f docker-compose.local.yml up -d --build
```

本地 Compose 只构建当前源码，不拉取 Docker Hub 镜像。最终发布验收应使用 `docker-compose.yml` 从 Docker Hub 全新拉取 `latest`。

## 持久化与端口

容器源码位于 `/app`，运行状态位于 `/app/state`。Compose 通过 `PERSONALITYRAG_STATE_DIR` 挂载宿主目录；模板只使用相对路径，不包含用户机器的绝对路径。

容器内部端口固定为：

- WebUI：`8765`
- 记忆库接入：`8766`

基础设置页中的两个端口在 Docker 模式下只读。需要修改宿主端口时，请编辑 `.env` 中的 `PERSONALITYRAG_WEBUI_HOST_PORT` 和 `PERSONALITYRAG_ACCESS_HOST_PORT`，然后重新执行 `docker compose up -d`。

默认 Embedding 地址为 `http://host.docker.internal:8001/v1`，只用于首次配置。模型运行在其他容器或远端时，可在 WebUI 中正常修改 Provider。

## 功能与平台一致性

Linux/Docker 版同步提供：

- 记忆、图谱、会话、Persona、Embedding 召回和 Rerank。
- OpenAI、Ollama、vLLM、Gemini 与 NVIDIA Embedding Provider。
- 可暂停、继续、停止、取消和异常恢复的持久化长任务。
- 安全断点、索引分段、任务前状态回滚和任务历史清理。
- GitHub Release 版本发现，以及 Docker 镜像原生的更新、回退和同版本重装。

Docker 镜像固定使用 64 位 CPython 3.12 作为构建与运行基线；这不改变 Windows 版 CPython 3.10+ x64 的兼容声明。

## 配置迁移

Docker 版可直接导入 Windows 版导出的 `.prag` 配置包。Provider、记忆库、密钥及其他设置会正常迁移；Docker 托管的监听地址和容器内部端口不会被 Windows 配置覆盖。Docker 导出的 `.prag` 包也可由 Windows 版读取。

## AstrBot 接入

- AstrBot 运行在宿主机：使用 `http://127.0.0.1:<映射后的接入端口>`。
- AstrBot 与 PersonalityRAG 位于同一 Docker 网络：使用 `http://personalityrag:8766`。

适配器仍需配置正确的记忆库 ID 和对应的 `psk-` 库级密钥。

## Docker Hub 标签

每个版本同时发布：

- `vX.Y.Z-amd64`
- `vX.Y.Z-arm64`
- 双架构 `vX.Y.Z`
- 与版本标签指向相同多架构 index 的 `latest`

## 开发检查

```bash
python -m pytest -q
docker compose -f docker-compose.local.yml config
docker build --target test -t personalityrag-linux:test .
docker run --rm personalityrag-linux:test
```

运行数据、配置、日志、数据库、索引、缓存和 `.env` 不进入 Git 或正式 Release ZIP。
