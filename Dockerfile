# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.13
ARG PYTHON_REGISTRY=docker.io/library
ARG NUMERICS_BASE_IMAGE=docker.io/vllm/vllm-openai:v0.27.0

FROM ${PYTHON_REGISTRY}/python:${PYTHON_VERSION}-slim-bookworm AS python-base

ARG COMFY_OMNI_BUILD_COMMIT
ARG COMFY_OMNI_BUILD_DIRTY

ENV COMFY_OMNI_BUILD_COMMIT=${COMFY_OMNI_BUILD_COMMIT} \
    COMFY_OMNI_BUILD_DIRTY=${COMFY_OMNI_BUILD_DIRTY} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN case "${COMFY_OMNI_BUILD_COMMIT}" in \
      *[!0-9a-f]*|'') echo 'COMFY_OMNI_BUILD_COMMIT must be a lowercase Git SHA' >&2; exit 2 ;; \
    esac \
    && test "${#COMFY_OMNI_BUILD_COMMIT}" -eq 40 \
    && case "${COMFY_OMNI_BUILD_DIRTY}" in 0|1) ;; *) exit 2 ;; esac

WORKDIR /workspace

FROM python-base AS development

COPY pyproject.toml setup.py README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY third_party ./third_party
COPY src ./src
RUN python -m pip install --no-cache-dir -e ".[dev]"
COPY . .

FROM development AS documentation

RUN python scripts/check_readme_sync.py \
    && python scripts/check_docker_policy.py \
    && python scripts/check_delivery_policy.py

FROM development AS quality

RUN python -m ruff format --check src tests scripts \
    && python -m ruff check src tests scripts \
    && python -m pytest -q --strict-markers \
    && python scripts/check_readme_sync.py \
    && python scripts/check_docker_policy.py \
    && python scripts/check_delivery_policy.py

FROM python-base AS package-builder

COPY pyproject.toml setup.py README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY third_party ./third_party
COPY src ./src
RUN python -m pip install --no-cache-dir build twine \
    && python -m build --outdir /dist \
    && python -m twine check /dist/*

FROM python-base AS package-check

COPY --from=package-builder /dist /dist
COPY scripts/check_installed_wheel.py /checks/check_installed_wheel.py
RUN python -m venv /venv \
    && /venv/bin/python -m pip install --no-cache-dir /dist/*.whl
WORKDIR /tmp/wheel-smoke
RUN /venv/bin/python /checks/check_installed_wheel.py

FROM python-base AS runtime

COPY --from=package-builder /dist/*.whl /tmp/wheels/
RUN python -m pip install --no-cache-dir /tmp/wheels/*.whl \
    && rm -r /tmp/wheels \
    && mkdir /work \
    && chown 65532:65532 /work
USER 65532:65532
WORKDIR /work
ENTRYPOINT ["comfy-omni"]
CMD ["--help"]

FROM ${NUMERICS_BASE_IMAGE} AS numerics-runtime

ARG COMFY_OMNI_BUILD_COMMIT
ARG COMFY_OMNI_BUILD_DIRTY

ENV COMFY_OMNI_BUILD_COMMIT=${COMFY_OMNI_BUILD_COMMIT} \
    COMFY_OMNI_BUILD_DIRTY=${COMFY_OMNI_BUILD_DIRTY} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN case "${COMFY_OMNI_BUILD_COMMIT}" in \
      *[!0-9a-f]*|'') echo 'COMFY_OMNI_BUILD_COMMIT must be a lowercase Git SHA' >&2; exit 2 ;; \
    esac \
    && test "${#COMFY_OMNI_BUILD_COMMIT}" -eq 40 \
    && case "${COMFY_OMNI_BUILD_DIRTY}" in 0|1) ;; *) exit 2 ;; esac

WORKDIR /opt/comfy-omni
COPY pyproject.toml setup.py README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY third_party ./third_party
COPY src ./src
RUN python3 -m pip install --no-cache-dir --no-deps . \
    && mkdir /work \
    && chown 65532:65532 /work

USER 65532:65532
WORKDIR /work
ENTRYPOINT ["python3"]
CMD ["-c", "from comfy_omni.conversion.numerics import regular_hadamard; print(regular_hadamard(4).shape)"]
