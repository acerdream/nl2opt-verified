# nl2opt 求解代码沙箱镜像
#
# 用途：在无网络、资源受限的容器内执行「IR → PuLP 代码」生成的求解脚本。
# 构建：docker build -t nl2opt-sandbox sandbox/
# 运行（由 src/nl2opt/sandbox.py 自动构造，此处仅示意）：
#   docker run --rm --network=none --memory=512m --cpus=1 --pids-limit=128 \
#     --read-only --tmpfs /tmp:rw,size=64m -e NL2OPT_TIMEOUT=25 \
#     -v <workdir>:/work:ro nl2opt-sandbox
FROM python:3.11-slim

# PyPI 源可覆盖（默认官方；网络受限时构建可传 --build-arg PIP_INDEX_URL=<镜像>）
ARG PIP_INDEX_URL=https://pypi.org/simple

# 仅安装求解依赖（PuLP 自带 CBC，故无需外部求解器）
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" "pulp>=2.7,<3"

# 非 root 运行，缩小容器内权限面
RUN useradd --create-home --uid 10001 sandbox

# 注意：此处**刻意不设** ENV PYTHONIOENCODING / PYTHONUTF8。
# 实测：容器内子进程 stdout 编码为 locale 默认（非 UTF-8 时如 gbk），而 runner.py 的
# ensure_ascii=False 会写出 UTF-8 字节；若在容器级强制 PYTHONIOENCODING=utf-8，
# 父进程会按 UTF-8 去解这批字节 → UnicodeDecodeError 被吞、stdout 静默变为 None
# → runner 崩溃且无信封（比不设更糟）。编码对齐改由 runner.py 的 subprocess
# 调用显式指定 encoding="utf-8", errors="replace" 完成，直接生效且无全局副作用。

COPY runner.py /opt/nl2opt/runner.py
WORKDIR /work

USER sandbox

# 容器内执行器：运行 /work/target.py 并输出结构化 JSON 信封
ENTRYPOINT ["python", "/opt/nl2opt/runner.py"]
