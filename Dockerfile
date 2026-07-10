FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# libgl1 / libglib2.0-0: runtime shared libs opencv-python links against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sudoku_chomper ./sudoku_chomper

# The locally trained models are baked in as-is, so container detection is
# byte-identical to the host. Training inside the image (the old setup) used the
# container's own fonts and drifted on noisy scans. Train on the host first:
#   python -m sudoku_chomper.train
#   python -m sudoku_chomper.train_style
COPY models/digit_svm.joblib models/style_svm.joblib ./models/

ENTRYPOINT ["python", "-m", "sudoku_chomper"]
