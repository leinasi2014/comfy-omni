#!/usr/bin/env bash
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly repository_root="$(cd -- "${script_dir}/.." && pwd)"
readonly action="${1:-quality}"
readonly python_version="${2:-3.13}"
readonly python_registry="${COMFY_OMNI_PYTHON_REGISTRY:-docker.io/library}"

case "${python_version}" in
  3.10|3.11|3.12|3.13) ;;
  *) echo "unsupported Python container version: ${python_version}" >&2; exit 2 ;;
esac
if [[ ! "${python_registry}" =~ ^[a-z0-9._:/-]+$ ]]; then
  echo "invalid COMFY_OMNI_PYTHON_REGISTRY: ${python_registry}" >&2
  exit 2
fi

cd "${repository_root}"
readonly source_commit="$(git rev-parse HEAD)"
source_dirty=0
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  source_dirty=1
fi
readonly source_dirty
readonly image="comfy-omni:${action}-${source_commit:0:12}-py${python_version}"

case "${action}" in
  docs) target=documentation ;;
  quality) target=quality ;;
  package) target=package-check ;;
  image|cli) target=runtime ;;
  *) echo "usage: $0 {docs|quality|package|image|cli} [3.10|3.11|3.12|3.13] [CLI_ARGS...]" >&2; exit 2 ;;
esac

docker build \
  --build-arg "PYTHON_VERSION=${python_version}" \
  --build-arg "PYTHON_REGISTRY=${python_registry}" \
  --build-arg "COMFY_OMNI_BUILD_COMMIT=${source_commit}" \
  --build-arg "COMFY_OMNI_BUILD_DIRTY=${source_dirty}" \
  --target "${target}" \
  --tag "${image}" \
  .

if [[ "${action}" == "cli" ]]; then
  if [[ "$#" -ge 2 ]]; then
    shift 2
  else
    shift 1
  fi
  if [[ "$#" -eq 0 ]]; then
    set -- --help
  fi
  docker run --rm \
    --network none \
    --read-only \
    --tmpfs /tmp:rw,noexec,nosuid,size=64m \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    "${image}" "$@"
fi
