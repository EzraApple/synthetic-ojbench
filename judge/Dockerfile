FROM python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd

ARG DMOJ_REVISION=f098cd3a49a60186d1fadde5132329ec5f4f2213
ARG OJBENCH_REVISION=5e94480b1e135b98855cf5bc81213c256aff5b17
LABEL org.synthetic-ojbench.dmoj-revision=$DMOJ_REVISION \
      org.synthetic-ojbench.ojbench-revision=$OJBENCH_REVISION

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DMOJ_PARALLEL=2
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ git ca-certificates libseccomp-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir setuptools==69.5.1 wheel==0.45.1 Cython==0.29.37 \
    && pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/DMOJ/judge-server.git@${DMOJ_REVISION}"
RUN git init /opt/ojbench \
    && git -C /opt/ojbench fetch --depth 1 https://github.com/He-Ren/OJBench.git "$OJBENCH_REVISION" \
    && git -C /opt/ojbench checkout --detach FETCH_HEAD \
    && pip install --no-cache-dir --no-build-isolation -e /opt/ojbench

WORKDIR /opt/judge
COPY run.py config.yaml runtime.yaml ./
RUN pip freeze > dependencies.txt && useradd --uid 1000 --create-home judge
USER judge
ENTRYPOINT ["python", "/opt/judge/run.py"]
