FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglfw3 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY . .
RUN python -m pip install --no-cache-dir -e ".[dev]"
CMD ["python", "scripts/run_pipeline.py"]

