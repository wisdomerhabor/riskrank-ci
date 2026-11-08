# Radon for Python 2-only source (Flask before ~2014). Radon 4.1.0 is used because Radon 5 dropped
# Python 2. Reads JSON [{"key", "source"}] on stdin, writes [{"key", "sloc", "complexity"}] to stdout.
FROM python:2.7-slim
RUN pip install --no-cache-dir "radon==4.1.0"
COPY measure.py /measure.py
ENTRYPOINT ["python", "/measure.py"]
