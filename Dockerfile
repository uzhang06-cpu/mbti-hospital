FROM python:3.11-slim

WORKDIR /app

# 先装依赖（利用缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷贝源码
COPY . .

ENV PORT=5001
EXPOSE 5001

# 与 Procfile 保持一致：eventlet worker 支持 WebSocket 长连接
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5001", "app:app"]
