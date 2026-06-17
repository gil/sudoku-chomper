FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# libgl1 / libglib2.0-0: runtime shared libs opencv-python links against.
# font packages: the digit classifier is trained from system fonts at build time
# (train.py globs /usr/share/fonts), so a varied set of Latin fonts must be present.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        fonts-dejavu-core \
        fonts-liberation \
        fonts-liberation2 \
        fonts-freefont-ttf \
        fonts-croscore \
        fonts-urw-base35 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sudoku_chomper ./sudoku_chomper

# Build models/digit_svm.joblib into the image using the installed fonts.
RUN python -m sudoku_chomper.train

# Build models/style_svm.joblib (printed vs handwritten; the handwriting detector that
# gates filtering). Needs the labeled REAL_SAMPLES scans (allowed past .dockerignore)
# and downloads MNIST.
COPY sample ./sample
RUN python -m sudoku_chomper.train_style

ENTRYPOINT ["python", "-m", "sudoku_chomper"]
