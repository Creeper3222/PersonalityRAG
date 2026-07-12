# PersonalityRAG Linux / Docker v0.1.0

PersonalityRAG 是独立运行的人格记忆 RAG 服务。本分支提供 Linux/Docker 发行版，与 Windows v0.1.0 保持数据库、API、WebUI、召回与配置包兼容，并增加 amd64/arm64 镜像和容器化持久部署。

- Windows 与项目主页：[Creeper3222/PersonalityRAG](https://github.com/Creeper3222/PersonalityRAG)
- AstrBot 适配器：[astrbot_plugin_personality_rag_adapter](https://github.com/Creeper3222/astrbot_plugin_personality_rag_adapter)
- Docker Hub：`138763327/personalityrag-linux`

项目采用 AGPL-3.0-only。存储、召回、图记忆和 LivingMemory DB v8 兼容设计受到 [astrbot_plugin_livingmemory](https://github.com/lxfight-s-Astrbot-Plugins/astrbot_plugin_livingmemory) 启发；完整版权与第三方说明见仓库根目录的许可证及声明文件。

## 快速启动

```bash
cp .env.example .env
docker compose up -d
```

默认地址：

- WebUI：`http://127.0.0.1:8765/`
- AstrBot/记忆库接入：`http://127.0.0.1:8766/`

Compose 默认拉取：

```text
138763327/personalityrag-linux:latest
```

首次启动会在挂载的状态目录生成随机 API Key、会话密钥、库级 PSK 派生密钥、配置和默认记忆库。请从容器日志读取首次 API Key，并妥善备份状态目录。

## 本地开发镜像

```bash
cp .env.example .env
docker compose -f docker-compose.local.yml up -d --build
```

本地 Compose 只构建当前源码，不拉取 Docker Hub 镜像。正式验收应使用 `docker-compose.yml` 强制拉取 `latest`。

## 持久化与端口

容器源码位于 `/app`，运行状态位于 `/app/state`。Compose 通过 `PERSONALITYRAG_STATE_DIR` 挂载宿主目录；模板使用相对路径，不包含任何本地绝对路径。

容器内部端口固定为：

- WebUI：`8765`
- 记忆库接入：`8766`

基础设置页中的两个端口在 Docker 模式下只读。需要改变宿主端口时，请修改 `.env` 中的 `PERSONALITYRAG_WEBUI_HOST_PORT` 和 `PERSONALITYRAG_ACCESS_HOST_PORT`，然后重新执行 `docker compose up -d`。

默认 Embedding 地址为 `http://host.docker.internal:8001/v1`，仅用于首次配置。模型运行在其它容器或远端时，可在 WebUI 中正常修改 Provider。

## 配置迁移

Docker 版可直接导入 Windows 版导出的 `.prag` 配置包。提供商、记忆库、密钥及其它设置正常迁移；Docker 托管的监听地址、WebUI 内部端口和接入内部端口会保留，不会被 Windows 配置覆盖。

Docker 导出的 `.prag` 包也可由 Windows 版读取。

## AstrBot 接入

- AstrBot 运行在宿主机：使用 `http://127.0.0.1:<映射后的接入端口>`。
- AstrBot 与 PersonalityRAG 位于同一 Docker 网络：使用 `http://personalityrag:8766`。

适配器仍需配置正确的记忆库 ID 和对应 `psk-` 库级密钥。

## Docker Hub 标签

每个版本同时发布：

- `vX.Y.Z-amd64`
- `vX.Y.Z-arm64`
- 双架构 `vX.Y.Z`
- 与版本标签相同 digest 的 `latest`

## 开发检查

```bash
python -m pytest -q
docker compose -f docker-compose.local.yml config
docker build --target test -t personalityrag-linux:test .
docker run --rm personalityrag-linux:test
```

运行数据、配置、日志、数据库、索引、缓存和 `.env` 不进入 Git 或正式 Release ZIP。
