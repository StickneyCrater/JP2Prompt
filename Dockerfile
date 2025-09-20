# ARM v7 Alpine Linux ベース FastAPI イメージ
FROM arm32v7/python:3.11-alpine

# メタデータ
LABEL maintainer="your-name@example.com"
LABEL description="ARM v7 Alpine Linux base image with FastAPI and Pydantic v1"
LABEL version="1.0"

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

# favicon.icoが存在することを確認（プレースホルダとして）
COPY favicon.ico favicon.ico

# 非rootユーザーを作成してセキュリティを向上
RUN addgroup -g 1001 -S appuser && \
    adduser -S -D -H -u 1001 -G appuser appuser && \
    chown -R appuser:appuser /app

USER appuser

# ポート8000を公開
EXPOSE 8091

# ヘルスチェック
HEALTHCHECK --interval=600s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8091/health || exit 1

# FastAPIアプリケーションを起動
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8091"]