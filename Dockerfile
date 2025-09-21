# ARM v7 Alpine Linux ベース FastAPI イメージ
FROM arm32v7/python:3.11-alpine

# メタデータ
LABEL maintainer="your-name@example.com"
LABEL description="ARM v7 Alpine Linux image with FastAPI, Ollama integration, and Forge API proxy"
LABEL version="2.0"

# 作業ディレクトリを設定
WORKDIR /app

# 必要最小限のシステムパッケージをインストール
RUN apk add --no-cache \
    curl \
    iputils \
    && rm -rf /var/cache/apk/*

# Pythonの依存関係をインストール（Pure Python版）
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# アプリケーションファイルをコピー
COPY . .

# 画像保存ディレクトリを作成
RUN mkdir -p /app/images

# 静的ファイル用ディレクトリを作成
RUN mkdir -p /app/static

# favicon.icoが存在することを確認（プレースホルダとして）
RUN touch favicon.ico

# 非rootユーザーを作成してセキュリティを向上
RUN addgroup -g 1001 -S appuser && \
    adduser -S -D -H -u 1001 -G appuser appuser && \
    chown -R appuser:appuser /app

USER appuser

# 環境変数設定
ENV TRANSLATE_HOST=192.168.2.199
ENV TRANSLATE_PORT=8091
ENV OLLAMA_HOST=192.168.2.197
ENV OLLAMA_PORT=11434
ENV FORGE_HOST=192.168.2.197
ENV FORGE_PORT=7860
ENV SAVE_DIR=/app/images

# ポート8091を公開
EXPOSE 8091

# ヘルスチェック
HEALTHCHECK --interval=600s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8091/health || exit 1

# FastAPIアプリケーションを起動
CMD ["uvicorn", "main:app", "--host", "192.168.2.199", "--port", "8091"]