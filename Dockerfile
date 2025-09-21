# ARM v7 Alpine Linux ベース FastAPI イメージ
FROM arm32v7/python:3.11-alpine

# メタデータ
LABEL maintainer="your-name@example.com"
LABEL description="ARM v7 Alpine Linux image with FastAPI, Ollama integration, and Forge API proxy"
LABEL version="2.0"

# 作業ディレクトリを設定
WORKDIR /app

# 必要最小限のシステムパッケージをインストール (sudo追加)
RUN apk add --no-cache \
    curl \
    iputils \
    util-linux \
    shadow \
    && rm -rf /var/cache/apk/*

# Pythonの依存関係をインストール（Pure Python版）
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# アプリケーションファイルをコピー
COPY . .

# ディレクトリを作成
RUN mkdir -p /app/images /app/config

# 管理用ユーザ rootuser (uid=1000, gid=1000)
RUN addgroup -g 1000 -S rootuser && \
    adduser -S -D -H -u 1000 -G rootuser -s /bin/sh rootuser && \
    echo "rootuser:rootuser123" | chpasswd && \
    echo "rootuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# アプリ用ユーザ appuser (uid=501, gid=100)
# gid=100 (everyone) は既に存在するので再作成せず -G 100 で指定する
RUN adduser -S -D -H -u 501 -G users -s /bin/sh appuser && \
    echo "appuser:appuser123" | chpasswd

# ディレクトリの所有権を QNAP 側 uid/gid (501:100) に合わせる
RUN chown -R 501:100 /app && \
    chmod -R 755 /app && \
    mkdir -p /tmp/mount_test && \
    chown 501:100 /tmp/mount_test && \
    chmod 755 /tmp/mount_test

# favicon.icoが存在することを確認（プレースホルダとして）
RUN touch favicon.ico && chown appuser:users favicon.ico

# 通常ユーザーとして実行
USER appuser

# 環境変数設定
ENV TRANSLATE_HOST=192.168.2.199
ENV TRANSLATE_PORT=8091
ENV OLLAMA_HOST=192.168.2.197
ENV OLLAMA_PORT=11434
ENV FORGE_HOST=192.168.2.197
ENV FORGE_PORT=7865
ENV SAVE_DIR=/app/images
ENV TRANSLATE_MODEL=brxce/stable-diffusion-prompt-generator:latest
ENV FORGE_MODEL=sd\\novaAnimeXL_ilV5b.safetensors

# ポート8091を公開
EXPOSE 8091

# ヘルスチェック
HEALTHCHECK --interval=600s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8091/health || exit 1

# FastAPIアプリケーションを起動
CMD ["sh", "-c", "uvicorn main:app --host $TRANSLATE_HOST --port $TRANSLATE_PORT"]