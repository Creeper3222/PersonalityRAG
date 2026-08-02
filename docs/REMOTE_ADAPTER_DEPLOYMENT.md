# Windows 远端适配器接入

本文档适用于单机、单实例、使用本机持久磁盘的 PersonalityRAG Windows
部署。它不代表多副本、共享 SQLite/FAISS、Kubernetes HA 或 mTLS 方案。

## 安全边界

PersonalityRAG 同时启动两个显式隔离的监听器：

| 监听器 | 默认地址 | 用途 | 建议可见范围 |
|---|---|---|---|
| WebUI | `http://127.0.0.1:8765` | 登录、设置、Provider、任务、文件和数据库管理 | 仅服务器本机；通过 RDP、SSH 隧道或 VPN 管理 |
| Adapter access | `http://127.0.0.1:8766` | 记忆库和知识库适配器协议 | 仅反向代理回源 |

公网只应开放 HTTPS `443`。不要把 `8765` 或 `8766` 直接暴露到公网，也不要
把 WebUI 代理到公开域名。接入监听器会把 `/`、静态资源、OpenAPI、登录、
设置、Provider、任务、文件及其它管理接口统一隐藏为 `404`；全局 API Key
和 WebUI 会话 Cookie 也不能突破该边界。

适配器接口必须使用目标库自己的密钥：

- `livingmemory_v8` 使用与库 ID 匹配的 `psk-` 密钥。
- `text_media_v1` 使用与库 ID 匹配的 `pkb-` 密钥。

接入监听器响应包含：

```text
X-PersonalityRAG-Surface: adapter-access
X-PersonalityRAG-Adapter-Protocol: 1
```

WebUI 监听器只返回 `X-PersonalityRAG-Surface: webui`。新版适配器会拒绝误连
WebUI；公网连接缺少接入面标识时也会拒绝继续发送库请求。

## 配置步骤

1. 保持 `config/config.json` 中两个监听器绑定到回环地址。不要改变为
   `0.0.0.0`。
2. 在“基础设置”中把“公网适配器 URL”设为实际 HTTPS 根地址，例如
   `https://memory.example.com` 或 `https://memory.example.com:8443`。
3. 公网地址不能带路径、用户名、密码、查询参数或片段。该配置仅用于 WebUI
   展示和复制，不会修改本地绑定地址或端口。
4. 使用 [Caddy 示例](examples/Caddyfile) 或
   [Nginx 示例](examples/nginx.personalityrag.conf) 只代理到
   `127.0.0.1:8766`。
5. 服务器防火墙仅对公网开放 `443/tcp`；限制或拒绝 `8765/tcp` 与
   `8766/tcp` 的外部入站。
6. 在适配器中填写公网根地址、目标库 ID 和库密钥。使用内部 CA 时填写 CA
   bundle 路径，不要关闭证书验证。

心跳允许最长 55 秒长轮询，因此反向代理的连接与响应头超时应至少为 70 秒。
代理不得自动跟随或改写到另一个 origin，也不要在访问日志中记录请求正文、
`Authorization` 或完整查询字符串。媒体签名参数具有临时访问能力。

## Caddy

复制 `docs/examples/Caddyfile`，替换域名后启动 Caddy。Caddy 默认自动申请和
续期证书；生产环境需要域名解析正确且 `80/443` 满足证书签发要求。示例没有
启用 Caddy access log，脱敏后的方法、路径、状态、耗时和请求 ID 由
PersonalityRAG 记录。

## Nginx

复制 `docs/examples/nginx.personalityrag.conf`，替换域名和证书路径。示例
访问日志使用 `$uri` 而不是 `$request_uri`，不会把媒体签名查询参数写入日志。
如果上游还有 CDN 或负载均衡器，同样应关闭 Authorization、请求正文和完整
查询字符串记录。

## 验证

从服务器外部网络验证：

```text
GET https://memory.example.com/api/v1/health
```

预期状态为 `200`，且响应包含 `adapter-access` 和协议版本标识。以下地址都
应返回 `404`，即使携带全局 API Key 或 WebUI Cookie：

```text
/
/static/
/docs
/openapi.json
/api/v1/settings
/api/v1/jobs
```

最后分别使用记忆适配器和知识适配器完成心跳、召回/检索及强制下线后手动
重连。只有在真实云服务器上完成 DNS、证书、防火墙与公网链路验证后，才能
声明已通过真实云端验收；本机反向代理容器测试只属于“类远端”验收。
