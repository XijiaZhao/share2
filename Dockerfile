# AudioAction model server.
#   docker build -t audioaction:1.0 .
#   docker run --gpus all -p 8025:8025 audioaction:1.0
# With --gpus it uses the GPU; without --gpus it runs on CPU (slower). Needs the NVIDIA Container
# Toolkit for --gpus.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app
RUN pip install --no-cache-dir soundfile pyyaml

COPY audioaction_a.pt audioaction_b.pt meta.json mel_filters.npz config.yml ./
COPY audio_frontend.py runtime.py serve.py ./

EXPOSE 8025
CMD ["python", "serve.py", "--config", "config.yml"]
