FROM python:3.12-slim
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/runner
COPY requirements.txt .
RUN pip install --no-cache-dir --require-hashes -r requirements.txt
COPY main.py .
USER 10001:10001
ENTRYPOINT ["python", "-I", "/opt/runner/main.py"]
