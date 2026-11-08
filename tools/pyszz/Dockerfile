# PySZZ v2 (Rosa et al., 2023), rebuilt for arm64 in 2026.
#
# The upstream Dockerfile no longer builds: it uses Debian buster (archived) and downloads srcML
# 1.0.0 from an address that now returns 404. Deviations from the published environment, all
# recorded in Chapter Four:
#   base image     ubuntu:22.04 (arm64)        upstream: python:3.7-slim-buster
#   python         3.10 (Ubuntu 22.04 system)  upstream: 3.7
#   srcML          1.2.0 (GitHub release)       upstream: 1.0.0
#   Java / RefactoringMiner  not installed     only RA-SZZ needs them; this project runs R-SZZ
#   python pins    unchanged — every upstream pin installs on 3.10 (verified 8 Nov 2026)
# PySZZ itself is unmodified and pinned to the commit below. GPL-3.0: run as a separate tool.

FROM ubuntu:22.04

ARG PYSZZ_COMMIT=67971581881993231b6918128436e1f06239c781
ARG SRCML_DEB=https://github.com/srcML/srcML/releases/download/v1.2.0/srcml_1.2.0-1_ubuntu22.04_arm64.deb

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git ca-certificates wget libarchive13 libcurl4 libxml2 libxslt1.1 \
    && wget -q -O /tmp/srcml.deb "$SRCML_DEB" && apt-get install -y /tmp/srcml.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/srcml.deb

WORKDIR /usr/src/app
RUN git clone https://github.com/grosa1/pyszz_v2.git . && git checkout "$PYSZZ_COMMIT"

COPY requirements-2026.txt .
RUN pip3 install --no-cache-dir -r requirements-2026.txt

RUN git config --global --add safe.directory '*'
ENTRYPOINT ["python3", "-u", "main.py"]
