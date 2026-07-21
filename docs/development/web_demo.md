# Web Demo — 开发文档

> Phase 11 (原 Phase 13)。参照 enterprise 项目 Docker 部署模式。
> 目标：单页 Web 应用，上传截图 → VLM + GNN → 对比叠加 bbox。

---

## 1. 架构总览

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  浏览器      │     │  Nginx       │     │  FastAPI     │
│  index.html  │────▶│  :8088       │────▶│  :8080       │
│  Canvas      │     │  静态文件     │     │  Inference   │
│  bbox overlay│     │  /api 代理    │     │  Pipeline    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                           ┌──────▼───────┐
                                           │  MySQL       │
                                           │  :3306       │
                                           │  推理历史     │
                                           └──────────────┘
```

### 1.1 三层结构（参照 enterprise 模式）

| 层 | 容器 | 端口 | 模式 |
|----|------|------|------|
| **前端** | nginx:alpine | `:8088` | 挂载 `index.html` 热更新 |
| **后端** | FastAPI (Python) | `:8080` | 挂载 wheel/源码热更新 |
| **数据库** | mysql:8.0 | `:3306` | 持久化卷 |

### 1.2 与 enterprise 的对应关系

| Enterprise 项目 | Web Demo 项目 | 说明 |
|----------------|--------------|------|
| Spring Boot JAR | FastAPI Python wheel | 后端 artifact |
| `Dockerfile.backend` | `Dockerfile.api` | 构建方式不同 |
| `index.html` | `web/index.html` | 前端单页 |
| `docker/nginx.conf` | `web/nginx.conf` | nginx 反向代理 |
| `docker-compose.yml` | `docker/docker-compose.yml` | 编排 |
| `init.sql` | `docker/init.sql` | 数据库初始化 |

---

## 2. 目录结构

```
bipartite-gnn-gui/
├── web/                      # 前端 + nginx
│   ├── index.html            # 单页应用（拖拽上传 + Canvas）
│   ├── nginx.conf            # /api → backend 代理
│   └── Dockerfile            # nginx:alpine 构建
├── api/                      # FastAPI 后端
│   ├── main.py               # FastAPI app + 路由
│   ├── inference.py          # InferencePipeline 封装
│   ├── requirements.txt      # FastAPI + uvicorn + pillow
│   ├── docker-entrypoint.sh  # 启动包装脚本
│   └── Dockerfile            # Python 轻量镜像
├── docker/                   # 部署编排 + 运维
│   ├── docker-compose.yml    # 三个服务
│   ├── .env.example          # 环境变量模板
│   ├── init.sql              # MySQL 初始化
│   └── demo                  # 运维 CLI 脚本
└── docs/development/
    └── web_demo.md           # ← 本文档
```

### 2.1 设计原则

- **前端不构建** — 纯 HTML + JS，无打包工具，`index.html` 直接挂载热更新
- **后端预编译** — `pip wheel .` 生成 wheel，`Dockerfile` 只 COPY，不 build（参照 enterprise JAR 模式）
- **服务独立重启** — 无 `depends_on` 耦合（参照 [docker-production-deployment](/docs/development/docker_patterns.md) 原则）
- **单页无路由** — SPA 不需要 `try_files`，直接 `index.html` 即可

---

## 3. 后端 API

### 3.1 FastAPI 应用 (`api/main.py`)

```python
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os, uuid, json
from inference import DemoInferencePipeline

app = FastAPI(title="GUI-GNN Demo")
pipeline = DemoInferencePipeline()

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    vlm_model: str = Form("qwen3-vl-flash"),
    api_key: str = Form(None),
):
    """上传截图 → VLM 检测 → GNN 修正 → 返回修正结果。"""
    img_bytes = await file.read()
    result = pipeline.run(img_bytes, vlm_model=vlm_model, api_key=api_key)
    return JSONResponse(result)

@app.post("/api/correct")
async def correct(
    file: UploadFile = File(...),
    vlm_json: str = Form(None),
):
    """上传截图 + VLM 预测 JSON → GNN 修正 → 返回对比结果。"""
    img_bytes = await file.read()
    vlm = json.loads(vlm_json) if vlm_json else None
    result = pipeline.correct(img_bytes, vlm_predictions=vlm)
    return JSONResponse(result)

@app.get("/api/health")
async def health():
    return {"status": "ok"}
```

### 3.2 DemoInferencePipeline 封装 (`api/inference.py`)

```python
class DemoInferencePipeline:
    """轻量封装 — 加载 checkpoint 后单次推理。"""

    def __init__(self, checkpoint: str = "/app/checkpoints/best_model.pt"):
        self.pipeline = InferencePipeline(checkpoint_path=checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, img_bytes: bytes, vlm_model: str, api_key: str) -> dict:
        """VLM → GNN 全流程。"""
        # 1. Save image temporarily
        # 2. Call VLM API for predictions
        # 3. Build constraint graph
        # 4. GNN inference
        # 5. Return corrected elements + overlay image
        ...

    def correct(self, img_bytes: bytes, vlm_predictions: list | None) -> dict:
        """只做 GNN 修正（已有 VLM 预测）。"""
        ...
```

### 3.3 API 响应格式

```json
{
  "status": "success",
  "elements": {
    "vlm": [
      {"bbox": [12, 34, 56, 78], "type": "button", "confidence": 0.85},
      ...
    ],
    "corrected": [
      {"bbox": [14, 36, 58, 76], "type": "button", "delta": [2, 2, 2, -2]},
      ...
    ]
  },
  "stats": {
    "vlm_count": 12,
    "gnn_count": 14,
    "corrections": 8,
    "new_proposals": 2,
    "inference_ms": 0.53
  },
  "overlay_b64": "data:image/png;base64,..."
}
```

---

## 4. Docker 部署

### 4.1 镜像引用（参照 enterprise 的 `image:` + `build:` 双声明）

```yaml
# docker/docker-compose.yml
services:
  db:
    image: mysql:8.0
    container_name: gui-demo-db
    restart: always
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD:-root123}
      MYSQL_DATABASE: gui_demo
      MYSQL_USER: gui_app
      MYSQL_PASSWORD: ${MYSQL_PASSWORD:-gui123}
    volumes:
      - mysql_data:/var/lib/mysql
      - ./init.sql:/docker-entrypoint-initdb.d/01-init.sql:ro
    ports:
      - "${DB_PORT:-3307}:3306"
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    image: gui-demo-api:latest
    build:
      context: ..
      dockerfile: api/Dockerfile
    container_name: gui-demo-api
    restart: always
    environment:
      DASHSCOPE_API_KEY: ${DASHSCOPE_API_KEY:-}
      MODEL_CHECKPOINT: /app/checkpoints/best_model.pt
      DB_HOST: db
      DB_PORT: 3306
      DB_NAME: gui_demo
      DB_USER: gui_app
      DB_PASSWORD: ${MYSQL_PASSWORD:-gui123}
    volumes:
      - ../checkpoints:/app/checkpoints:ro
      - uploads_data:/app/uploads
    ports:
      - "${API_PORT:-8080}:8080"
    # 无 depends_on — 独立重启（参照 enterprise 原则）

  frontend:
    image: gui-demo-frontend:latest
    build:
      context: ..
      dockerfile: web/Dockerfile
    container_name: gui-demo-frontend
    restart: always
    volumes:
      - ../web/index.html:/usr/share/nginx/html/index.html:ro
    ports:
      - "${FRONTEND_PORT:-8088}:80"

volumes:
  mysql_data:
  uploads_data:
```

### 4.2 启动包装脚本（参照 enterprise crash-loop wrapper）

```dockerfile
# api/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 预编译 wheel（参照 JAR 模式）
COPY api/dist/bipartite_gnn_gui_demo-*.whl .
RUN pip install --no-cache-dir *.whl && rm *.whl

COPY api/inference.py .
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 启动包装脚本 — 保持容器 Up
COPY api/docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -sf http://localhost:8080/api/health || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
```

```bash
#!/bin/bash
# api/docker-entrypoint.sh — 保持容器 Up（参照 enterprise 模式）
set -e
echo "[entrypoint] Starting FastAPI server..."

while true; do
  uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1
  echo "[entrypoint] Server exited with code $?, restarting in 3s..."
  sleep 3
done
```

### 4.3 Nginx 配置

```nginx
# web/nginx.conf
upstream api {
    server api:8080;
}

server {
    listen 80;
    server_name _;
    client_max_body_size 20M;

    # 静态文件
    location / {
        root /usr/share/nginx/html;
        index index.html;
    }

    # API 代理
    location /api/ {
        proxy_pass http://api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
    }
}
```

### 4.4 前端 Dockerfile

```dockerfile
# web/Dockerfile
FROM nginx:alpine

RUN rm -f /etc/nginx/conf.d/default.conf

COPY web/index.html /usr/share/nginx/html/
COPY web/nginx.conf /etc/nginx/conf.d/demo.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -sf http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
```

---

## 5. MySQL 初始化

```sql
-- docker/init.sql
CREATE TABLE IF NOT EXISTS inference_history (
    id          VARCHAR(36) PRIMARY KEY,
    filename    VARCHAR(255) NOT NULL,
    vlm_model   VARCHAR(64),
    elem_before INT,
    elem_after  INT,
    corrections INT,
    proposals   INT,
    inference_ms FLOAT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vlm_predictions (
    id            VARCHAR(36) PRIMARY KEY,
    history_id    VARCHAR(36) NOT NULL,
    bbox_json     JSON NOT NULL,
    type          VARCHAR(32),
    confidence    FLOAT,
    is_corrected  BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (history_id) REFERENCES inference_history(id)
);
```

---

## 6. 开发工作流

### 6.1 首次启动

```bash
# 1. 预编译后端 wheel
cd api && python -m pip wheel . -w dist

# 2. 启动所有服务
cd ../docker
cp .env.example .env
docker compose up -d

# 3. 验证
demo status
```

### 6.2 热更新（用 CLI）

```bash
# 更新前端
demo update -f

# 更新后端（先编译 wheel，再运行）
cd api && pip wheel . -w dist && cd ..
demo update -b

# 一键全部更新
demo update
```

### 6.3 构建模式说明

| 环境 | 构建方式 | 说明 |
|------|---------|------|
| **开发** | Wheel 预编译 + `demo start` | 10 秒部署 |
| **演示** | 同上 | checkpoint 预下载 |
| **部署** | 同 enterprise 离线包 | `docker save` → tar → USB |

### 6.4 `.env` 配置

```bash
# docker/.env.example
MYSQL_ROOT_PASSWORD=root123
MYSQL_PASSWORD=gui123
DASHSCOPE_API_KEY=sk-xxx         # Qwen3-VL Flash API key
API_PORT=8080
FRONTEND_PORT=8088
DB_PORT=3307
```

---

## 7. 前端要点

### 7.1 页面布局

```
┌─────────────────────────────────────┐
│  [Upload Area]   拖拽/点击上传截图    │
├──────────┬──────────────────────────┤
│  Before  │         After            │
│  (VLM)   │    (VLM + GNN)          │
│          │                          │
│  ┌────┐  │    ┌────┐               │
│  │ ✅  │  │    │ ✅  │               │
│  └────┘  │    └────┘               │
│  ┌────┐  │    ┌────┐  ┌──────┐    │
│  │ ❌  │  │    │ ✅  │  │ 新增  │    │
│  └────┘  │    └────┘  └──────┘    │
├──────────┴──────────────────────────┤
│  统计: VLM 12 个, GNN 14 个, 修正 8 个  │
└─────────────────────────────────────┘
```

### 7.2 Canvas 叠加

- 底色：截图原图
- VLM bbox：**红色**半透明（RGBA `255,0,0,0.3`）
- GNN 修正后 bbox：**绿色**半透明（RGBA `0,255,0,0.3`）
- GNN 新增提议：**蓝色**虚线
- 标签：每类元素用不同形状标记（圆点 = button, 方块 = text, 三角 = icon）

### 7.3 前端无框架

- 纯原生 JS，无 React/Vue
- CSS：轻量 flexbox 布局
- 拖拽上传：原生 `dragenter`/`dragover`/`drop` 事件
- Canvas 绘制：`ctx.drawImage` + `ctx.strokeRect` + `ctx.fillText`

---

## 8. 关键模式对照（Enterprise → GNN Demo）

| 模式 | Enterprise | Web Demo |
|------|-----------|----------|
| 后端 artifact | `enterprise-mvp-0.1.0-SNAPSHOT.jar` | `bipartite_gnn_gui_demo-*.whl` |
| 构建方式 | 预编译 JAR | 预编译 wheel |
| 启动包装 | `docker-entrypoint.sh` | `docker-entrypoint.sh` |
| 前端 | nginx + `index.html` | nginx + `index.html` |
| API 代理 | `/api/` → backend:8080 | `/api/` → api:8080 |
| 独立重启 | 无 `depends_on` | 无 `depends_on` |
| DB | MySQL 8.0 | MySQL 8.0 |
| 热更新 | `docker cp` | volume 挂载 + CLI |
| 运维 CLI | `enterprise` CLI | `demo` CLI |
| 健康检查 | `HEALTHCHECK CMD curl` | `HEALTHCHECK CMD curl` |
| 配置 | `.env` | `.env` |

---

## 9. 依赖清单

### 9.1 Python 依赖

```
# api/requirements.txt
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6
pillow>=10.0.0
torch>=2.1.0
torch-geometric>=2.4.0
numpy>=1.24.0
pydantic>=2.5.0
```

### 9.2 Docker 依赖

| 镜像 | 用途 | 大小 |
|------|------|------|
| `python:3.11-slim` | FastAPI 后端 | ~150MB |
| `nginx:alpine` | 前端静态服务 | ~25MB |
| `mysql:8.0` | 数据库 | ~600MB |
| `eclipse-temurin:17-jre` | （可选）JRE | ~200MB |

### 9.3 模型依赖

| 文件 | 来源 | 大小 |
|------|------|------|
| `checkpoints/best_model.pt` | 训练产出 | ~5MB |
| `checkpoints/confidence_scoring/` | Phase 4.8 | ~5MB |

---

## 10. 开发路线

| # | 任务 | 预计工时 | 前置 |
|---|------|---------|------|
| 11.1 | FastAPI 后端 (`api/main.py` + `api/inference.py`) | 2h | 训练 checkpoint |
| 11.2 | 前端 (`web/index.html` + Canvas 叠加) | 3h | 11.1 API 定义 |
| 11.3 | Docker 编排 (`docker/docker-compose.yml`) | 1h | 11.1, 11.2 |
| 11.4 | MySQL 集成 + 历史记录 | 1h | 11.1 |
| 11.5 | 运维 CLI (`docker/demo`) | 1h | 11.3 |
| 11.6 | 端到端测试 | 1h | 11.3, 11.4 |
| 11.7 | 文档 (`README.md` + 部署指南) | 1h | 11.6 |

---

## 11. 注意事项

1. **checkpoint 体积** — `best_model.pt` 约 5MB，直接挂载 volumes 即可
2. **Postman 测试** — 上传截图 + VLM JSON 时，`vlm_json` 字段用 Form 传递（非 JSON body）
3. **CORS** — nginx 转发时 `proxy_set_header` 已处理跨域
4. **文件清理** — 定期清理 `uploads/` 中的临时截图
5. **VLM API key** — `DASHSCOPE_API_KEY` 必须配置，否则 VLM 调用失败
6. **MySQL 端口冲突** — 默认 `:3307` 映射，避免与本机 MySQL `:3306` 冲突
7. **`set -u` 空数组** — CLI 中 `for x in "${arr[@]}"` 在空数组下会 crash，需用 `[ ${#arr[@]} -gt 0 ]` 守卫
8. **`read` 在非 TTY** — CLI 中所有 `read -p` 都加 `|| true`，否则管道输入时 crash

---

## 12. 运维 CLI

参照 enterprise 项目的 `enterprise` CLI 模式，Web Demo 需要一个统一的运维脚本 `demo`，包装 docker compose 操作 + 健康检查 + 热更新。

### 12.1 CLI 命令一览

```bash
# 安装后全局可用
demo start          # docker compose up -d + 等待健康
demo stop           # docker compose down
demo restart        # 重启后端服务（热更新后）
demo status         # 容器状态 + 健康检查汇总
demo logs [-f]      # 查看日志（auto-detect 后端/前端/DB）
demo update [-b|-f] # 热更新后端 wheel 或前端 index.html
demo check-api      # VLM API 连通性检查
```

### 12.2 命令详解

| 命令 | 功能 | 典型输出 |
|------|------|---------|
| `demo start` | 启动全部服务，等待 API 健康（最长 60s） | `Starting services... → API healthy ✅` |
| `demo stop` | 停止全部服务 | `Stopping... → gui-demo-api removed` |
| `demo restart` | 仅重启 API 容器（热更新后） | `Restarting API... → API healthy ✅` |
| `demo status` | 容器状态 + 三层健康检查 | `gui-demo-api Up ✅ / gui-demo-frontend Up ✅ / gui-demo-db Up ✅` |
| `demo logs [svc]` | 查看日志，默认后端 | `tail -f api logs...` |
| `demo update [-b/-f]` | 热更新后端 wheel 或前端 HTML | `Backend updated ✅` / `Frontend updated ✅` |
| `demo check-api` | 诊断 VLM API 连通性 | `DashScope key: ✅ configured` |

### 12.3 CLI 完整实现 (`docker/demo`)

```bash
#!/bin/bash
# demo — Web Demo 运维 CLI
set -euo pipefail

# ── 自动检测项目根目录 ──
find_project_root() {
  for dir in "${DEMO_HOME:-}" /opt/bipartite-gnn-gui "$(cd "$(dirname "$0")/.." && pwd)"; do
    [ -f "$dir/docker/docker-compose.yml" ] && { echo "$dir"; return 0; }
  done
  echo "Error: project root not found" >&2; exit 1
}
PROJECT_ROOT=$(find_project_root)
COMPOSE_DIR="$PROJECT_ROOT/docker"

# ── 容器名 ──
API_CONT="gui-demo-api"
FE_CONT="gui-demo-frontend"
DB_CONT="gui-demo-db"

# ── 读取 .env ──
if [ -f "$COMPOSE_DIR/.env" ]; then
  source "$COMPOSE_DIR/.env" 2>/dev/null || true
fi
API_PORT="${API_PORT:-8080}"
FE_PORT="${FRONTEND_PORT:-8088}"

# ── 颜色 ──
green() { printf "\033[32m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }
warn()  { printf "\033[33m%s\033[0m\n" "$*"; }

# ══════════════════════════════════════
#  命令实现
# ══════════════════════════════════════

cmd_start() {
  echo "→ Starting services..."
  docker compose -f "$COMPOSE_DIR/docker-compose.yml" up -d

  # 等待健康（最多 60 秒）— 不阻塞 start 完成
  echo -n "→ Waiting for API health..."
  local ok=false
  for i in $(seq 1 20); do
    if curl -sf "http://localhost:$API_PORT/api/health" > /dev/null 2>&1; then
      green " OK (${i}s)"
      ok=true
      break
    fi
    echo -n "."; sleep 3
  done
  $ok || red " timeout"

  cmd_status
}

cmd_stop() {
  docker compose -f "$COMPOSE_DIR/docker-compose.yml" down "$@"
  green "stopped"
}

cmd_restart() {
  echo "→ Restarting API..."
  docker compose -f "$COMPOSE_DIR/docker-compose.yml" restart api
  sleep 5
  if curl -sf "http://localhost:$API_PORT/api/health" > /dev/null; then
    green "API healthy"
  else
    red "API unhealthy!"
  fi
}

cmd_status() {
  echo "── Container Status ──"
  docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" \
    --filter "name=gui-demo" 2>/dev/null || echo "(no gui-demo containers running)"

  echo ""
  echo "── Health Checks ──"

  # API
  if curl -sf "http://localhost:$API_PORT/api/health" > /dev/null 2>&1; then
    green "  API       :$API_PORT  ✅ healthy"
  else
    red "  API       :$API_PORT  ❌ unreachable"
  fi

  # Frontend
  if curl -sf "http://localhost:$FE_PORT/" > /dev/null 2>&1; then
    green "  Frontend  :$FE_PORT  ✅ serving"
  else
    red "  Frontend  :$FE_PORT  ❌ unreachable"
  fi

  # DB
  if docker exec "$DB_CONT" mysqladmin ping -u root \
    -p"${MYSQL_ROOT_PASSWORD:-root123}" --silent 2>/dev/null; then
    green "  Database  :3306  ✅ accepting"
  else
    red "  Database  :3306  ❌ unreachable"
  fi
}

cmd_logs() {
  local service="${1:-}"
  [ -z "$service" ] && service="api"  # 默认看后端日志
  docker compose -f "$COMPOSE_DIR/docker-compose.yml" logs "$@" "$service"
}

cmd_update() {
  # 解析 flags（参照 enterprise CLI 模式）
  UPDATE_API=false; UPDATE_FE=false
  while [ $# -gt 0 ]; do
    case "$1" in
      -b|--backend) UPDATE_API=true ;;
      -f|--frontend) UPDATE_FE=true ;;
      *) warn "Unknown flag: $1"; exit 1 ;;
    esac; shift
  done
  HAS_FLAGS=false; $UPDATE_API && HAS_FLAGS=true; $UPDATE_FE && HAS_FLAGS=true

  # ── 更新后端 wheel ──
  do_api=false
  if [ "$UPDATE_API" = true ]; then do_api=true
  elif [ "$HAS_FLAGS" = false ]; then
    echo -n "  Update API backend? [Y/n] "; read -r ans || true
    [ "$ans" != "n" ] && do_api=true
  fi

  if [ "$do_api" = true ]; then
    WHEEL_OPTS=()
    for dir in "$PROJECT_ROOT/api/dist" "$PROJECT_ROOT/dist"; do
      _w=$(ls "$dir"/bipartite_gnn_gui_demo-*.whl 2>/dev/null | head -1) || true
      [ -n "$_w" ] && WHEEL_OPTS+=("$_w")
    done

    case ${#WHEEL_OPTS[@]} in
      0) warn "  No wheel found — run 'cd api && pip wheel . -w dist' first" ;;
      1) src="${WHEEL_OPTS[0]}" ;;
      *)
        echo "  Choose wheel:"
        for i in "${!WHEEL_OPTS[@]}"; do echo "    [$((i+1))] ${WHEEL_OPTS[$i]}"; done
        echo -n "  [1-${#WHEEL_OPTS[@]}] (default 1): "; read -r sel || true
        sel=${sel:-1}; src="${WHEEL_OPTS[$((sel-1))]}"
        ;;
    esac

    if [ -n "${src:-}" ]; then
      echo "→ Copying wheel: $(basename "$src")"
      docker cp "$src" "$API_CONT:/app/"
      docker exec "$API_CONT" pip install --no-cache-dir \
        --force-reinstall "/app/$(basename "$src")" > /dev/null
      echo "→ Restarting API..."
      docker restart "$API_CONT"
      sleep 5
      if curl -sf "http://localhost:$API_PORT/api/health" > /dev/null; then
        green "  API updated ✅"
      else
        red "  API unhealthy after update!"
      fi
    fi
  fi

  # ── 更新前端 index.html ──
  do_fe=false
  if [ "$UPDATE_FE" = true ]; then do_fe=true
  elif [ "$HAS_FLAGS" = false ]; then
    echo -n "  Update frontend index.html? [Y/n] "; read -r ans || true
    [ "$ans" != "n" ] && do_fe=true
  fi

  if [ "$do_fe" = true ]; then
    src="$PROJECT_ROOT/web/index.html"
    if [ -f "$src" ]; then
      docker cp "$src" "$FE_CONT:/usr/share/nginx/html/index.html"
      docker exec "$FE_CONT" nginx -s reload > /dev/null 2>&1
      green "  Frontend updated ✅"
    else
      warn "  $src not found"
    fi
  fi

  echo ""
  cmd_status
}

cmd_check_api() {
  echo "── VLM API Diagnostic ──"

  # 1. 检查 API key
  KEY="${DASHSCOPE_API_KEY:-}"
  if [ -z "$KEY" ]; then
    KEY=$(docker exec "$API_CONT" env 2>/dev/null | \
      grep DASHSCOPE_API_KEY | sed 's/.*=//') || true
  fi
  if [ -z "$KEY" ] || [ "$KEY" = "sk-xxx" ]; then
    red "  DASHSCOPE_API_KEY: 未配置或占位符"
    echo "    → 编辑 docker/.env 设置正确密钥"
    return
  fi
  green "  DASHSCOPE_API_KEY: ✅ 已配置 (长度 ${#KEY})"

  # 2. 测试 API 连通性
  echo -n "  Testing DashScope API..."
  STATUS=$(curl -sf -o /dev/null -w "%{http_code}" \
    -X POST "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions" \
    -H "Authorization: Bearer ${KEY:0:8}..." \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3-vl-flash","messages":[{"role":"user","content":[{"type":"text","text":"ping"}]}],"max_tokens":1}' \
    2>/dev/null || echo "000")
  case "$STATUS" in
    000) red " ❌ 无法连接 (网络/DNS)" ;;
    400) warn " ⚠ API 可达但请求异常 (HTTP 400)" ;;
    401) red " ❌ API key 无效 (HTTP 401)" ;;
    200) green " ✅ 连通 (HTTP 200)" ;;
    *)   green " ✅ 响应 $STATUS" ;;
  esac
}

# ══════════════════════════════════════
#  main
# ══════════════════════════════════════

case "${1:-}" in
  start)      shift; cmd_start "$@" ;;
  stop)       shift; cmd_stop "$@" ;;
  restart)    cmd_restart ;;
  status)     cmd_status ;;
  logs)       shift; cmd_logs "$@" ;;
  update)     shift; cmd_update "$@" ;;
  check-api)  cmd_check_api ;;
  *)
    echo "Usage: demo <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start           Start all services"
    echo "  stop            Stop all services"
    echo "  restart         Restart API only"
    echo "  status          Show container + health status"
    echo "  logs [service]  Tail logs (api|frontend|db)"
    echo "  update [-b|-f]  Hot-update backend or frontend"
    echo "  check-api       Diagnose VLM API connectivity"
    ;;
esac
```

### 12.4 安装

```bash
# 复制到全局路径
sudo cp docker/demo /usr/local/bin/demo
sudo chmod +x /usr/local/bin/demo

# 验证
demo status
```

### 12.5 使用示例

```bash
# 首次启动
demo start

# 开发中热更新前端
# 修改 web/index.html 后：
demo update -f

# 开发中热更新后端
# cd api && pip wheel . -w dist 后：
demo update -b

# 排查 VLM API
demo check-api

# 查看日志
demo logs api -f
demo logs frontend
```

### 12.6 `set -u` 安全模式说明

CLI 脚本使用 `set -euo pipefail`，需要特别注意以下 bash 陷阱：

| 模式 | 不安全写法 | 安全写法 |
|------|-----------|---------|
| 空数组遍历 | `for x in "${arr[@]}"` | `[ ${#arr[@]} -gt 0 ] && for x in "${arr[@]}"` |
| 非 TTY read | `read -p "..." var` | `read -p "..." var \|\| true` |
| ls 无匹配 | `jar=$(ls *.jar)` | `jar=$(ls *.jar 2>/dev/null \| head -1) \|\| true` |
| 未定义变量 | `for dir in $VAR /opt` | `for dir in "${VAR:-}" /opt` |
