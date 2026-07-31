#!/usr/bin/env bash
# KVarN one-click installer and verifier
#
# Default:
#   KVarN source: /data/wanglin/KVarN
#   runtime smoke model: /data/wanglin/models/Qwen3-4B
#
# Typical use:
#   conda activate kvarn-smoke
#   bash kvarn_oneclick.sh
#
# Check only:
#   bash kvarn_oneclick.sh --check-only
#
# Explicit local wheel:
#   bash kvarn_oneclick.sh --wheel /path/to/vllm-....whl
#
# Force source build:
#   bash kvarn_oneclick.sh --mode source

set -Eeuo pipefail
IFS=$'\n\t'

ROOT="${KVARN_ROOT:-/data/wanglin/KVarN}"
MODE="${KVARN_INSTALL_MODE:-auto}"
EXPLICIT_WHEEL="${KVARN_WHEEL:-}"
BASE_VERSION="${KVARN_BASE_VLLM_VERSION:-}"
VLLM_COMMIT="${KVARN_VLLM_COMMIT:-}"
WHEEL_VARIANT="${KVARN_WHEEL_VARIANT:-}"
DONOR_ROOT="${KVARN_DONOR_VLLM_ROOT:-}"
SMOKE_MODEL="${KVARN_SMOKE_MODEL:-/data/wanglin/models/Qwen3-4B}"
RUN_SMOKE="auto"
CHECK_ONLY=0
ASSUME_YES=0
MAX_JOBS_VALUE="${MAX_JOBS:-8}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
INSTALL_STRATEGY=""
STATIC_STATUS="NOT_RUN"
RUNTIME_STATUS="NOT_RUN"
LAST_ERROR=""

usage() {
  cat <<'EOF'
KVarN one-click installer and verifier

Usage:
  bash kvarn_oneclick.sh [options]

Options:
  --root PATH             KVarN source root
                          default: /data/wanglin/KVarN
  --mode MODE             auto | precompiled | donor | source
                          default: auto
  --wheel PATH_OR_URL     Explicit compatible vLLM wheel
  --base-version VERSION  Base vLLM tag, e.g. v0.23.0
  --commit SHA40          Exact upstream vLLM commit for wheel lookup
  --variant NAME          Preferred wheel variant, e.g. cu126 or cu129
  --donor-root PATH       Existing working vLLM source/install tree
  --smoke-model PATH      Local model for an actual KVarN runtime test
  --skip-runtime-smoke    Only perform import/static verification
  --require-runtime-smoke Fail if the local model is unavailable or smoke fails
  --check-only            Do not install; only verify current environment
  --max-jobs N            Parallel jobs for local source compilation
  -y, --yes               Do not ask before replacing current vLLM installation
  -h, --help              Show help

Environment-variable equivalents:
  KVARN_ROOT
  KVARN_INSTALL_MODE
  KVARN_WHEEL
  KVARN_BASE_VLLM_VERSION
  KVARN_VLLM_COMMIT
  KVARN_WHEEL_VARIANT
  KVARN_DONOR_VLLM_ROOT
  KVARN_SMOKE_MODEL
  MAX_JOBS

Install strategy in auto mode:
  1. Explicit wheel, when provided
  2. Exact-commit wheel from wheels.vllm.ai, filtered by CPU architecture
  3. Binary carrier generated from an existing compatible vLLM installation
  4. Full local source compilation, only when build prerequisites exist

The script never intentionally falls back to the current nightly wheel.
EOF
}

while (($#)); do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --wheel) EXPLICIT_WHEEL="$2"; shift 2 ;;
    --base-version) BASE_VERSION="$2"; shift 2 ;;
    --commit) VLLM_COMMIT="$2"; shift 2 ;;
    --variant) WHEEL_VARIANT="$2"; shift 2 ;;
    --donor-root) DONOR_ROOT="$2"; shift 2 ;;
    --smoke-model) SMOKE_MODEL="$2"; shift 2 ;;
    --skip-runtime-smoke) RUN_SMOKE="no"; shift ;;
    --require-runtime-smoke) RUN_SMOKE="required"; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --max-jobs) MAX_JOBS_VALUE="$2"; shift 2 ;;
    -y|--yes) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1"; usage; exit 2 ;;
  esac
done

case "$MODE" in
  auto|precompiled|donor|source) ;;
  *) echo "[ERROR] --mode must be auto, precompiled, donor, or source"; exit 2 ;;
esac

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] python was not found. Activate the kvarn-smoke environment first."
  exit 1
fi

ROOT="$(readlink -f "$ROOT" 2>/dev/null || printf '%s' "$ROOT")"
LOG_DIR="${KVARN_LOG_DIR:-$ROOT/.install-logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/kvarn_install_${STAMP}.log"
REPORT_FILE="$LOG_DIR/kvarn_install_${STAMP}.report"

exec > >(tee -a "$LOG_FILE") 2>&1

info() { printf '\n[%s] %s\n' "INFO" "$*"; }
warn() { printf '\n[%s] %s\n' "WARN" "$*" >&2; }
die()  { printf '\n[%s] %s\n' "ERROR" "$*" >&2; exit 1; }

on_error() {
  local code=$?
  local line=${BASH_LINENO[0]:-unknown}
  LAST_ERROR="command failed at line ${line}, exit=${code}"
  printf '\n[ERROR] %s\n' "$LAST_ERROR" >&2
  printf '[ERROR] Full log: %s\n' "$LOG_FILE" >&2
  write_report || true
  exit "$code"
}
trap on_error ERR

write_report() {
  {
    echo "timestamp=$STAMP"
    echo "root=$ROOT"
    echo "python=$PYTHON_BIN"
    echo "mode=$MODE"
    echo "strategy=${INSTALL_STRATEGY:-none}"
    echo "base_vllm_version=${BASE_VERSION:-unknown}"
    echo "vllm_commit=${VLLM_COMMIT:-unknown}"
    echo "wheel_variant=${WHEEL_VARIANT:-unknown}"
    echo "static_check=$STATIC_STATUS"
    echo "runtime_smoke=$RUNTIME_STATUS"
    echo "log=$LOG_FILE"
    echo "last_error=${LAST_ERROR:-none}"
  } > "$REPORT_FILE"
}

section() {
  printf '\n============================================================\n'
  printf '%s\n' "$*"
  printf '============================================================\n'
}

confirm_replacement() {
  if (( ASSUME_YES )); then
    return
  fi
  if [[ ! -t 0 ]]; then
    warn "Non-interactive input detected; proceeding without confirmation."
    return
  fi
  echo
  echo "This will replace the vLLM package registration in the current Python environment."
  read -r -p "Continue? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || die "Cancelled."
}

preflight() {
  section "1/8 Preflight"

  [[ "$(uname -s)" == "Linux" ]] || die "vLLM/KVarN GPU installation requires Linux."
  [[ -d "$ROOT" ]] || die "KVarN root does not exist: $ROOT"
  [[ -f "$ROOT/setup.py" ]] || die "Missing $ROOT/setup.py"
  [[ -f "$ROOT/pyproject.toml" ]] || die "Missing $ROOT/pyproject.toml"
  [[ -d "$ROOT/vllm" ]] || die "Missing $ROOT/vllm"

  local arch
  arch="$(uname -m)"
  case "$arch" in
    x86_64|aarch64) ;;
    *) die "Unsupported CPU architecture for this script: $arch" ;;
  esac

  unset PYTHONPATH || true
  export PYTHONNOUSERSITE=1
  export PIP_DISABLE_PIP_VERSION_CHECK=1

  echo "KVarN root       : $ROOT"
  echo "Python           : $PYTHON_BIN"
  echo "Python prefix    : $("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')"
  echo "Conda env        : ${CONDA_DEFAULT_ENV:-not-active}"
  echo "CPU architecture : $arch"
  echo "Install mode     : $MODE"
  echo "Log              : $LOG_FILE"

  if [[ "${CONDA_DEFAULT_ENV:-}" != "kvarn-smoke" ]]; then
    warn "Current conda environment is '${CONDA_DEFAULT_ENV:-none}', not 'kvarn-smoke'."
    warn "The script continues because the selected Python may still be correct."
  fi

  if ! grep -Rqs --include='*.py' 'kvarn_k4v2_g128' "$ROOT/vllm"; then
    die "The source tree does not contain kvarn_k4v2_g128. This is not a complete KVarN source tree."
  fi

  local backend_count
  backend_count="$(find "$ROOT/vllm" -type f -iname '*kvarn*.py' | wc -l | tr -d ' ')"
  (( backend_count > 0 )) || die "No KVarN Python backend files were found under $ROOT/vllm."

  echo "KVarN dtype source check : PASS"
  echo "KVarN backend files      : $backend_count"

  if command -v git >/dev/null 2>&1 && git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "KVarN Git HEAD           : $(git -C "$ROOT" rev-parse HEAD)"
    if [[ -n "$(git -C "$ROOT" status --short)" ]]; then
      warn "The KVarN tree has local modifications. The script will not reset or clean them."
    fi
  else
    warn "KVarN root is not a readable Git checkout. Exact version detection will rely on README/source metadata."
  fi
}

inspect_torch() {
  section "2/8 Python, PyTorch, and CUDA"

  "$PYTHON_BIN" - <<'PY'
import json
import platform
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch import failed: {exc}")

data = {
    "python": sys.version.split()[0],
    "executable": sys.executable,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "torch": torch.__version__,
    "torch_path": torch.__file__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
}
if torch.cuda.is_available():
    data["gpu_0"] = torch.cuda.get_device_name(0)
print(json.dumps(data, indent=2, ensure_ascii=False))

if torch.version.cuda is None:
    raise SystemExit("This PyTorch build has no CUDA support.")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch.")
PY
}

normalize_version() {
  local value="$1"
  value="${value#v}"
  if [[ "$value" =~ ^([0-9]+\.[0-9]+\.[0-9]+) ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  else
    printf '%s' "$value"
  fi
}

detect_base_version() {
  section "3/8 Detect KVarN base vLLM version"

  if [[ -z "$BASE_VERSION" ]]; then
    BASE_VERSION="$(
      grep -Eio 'built on[[:space:]]+vLLM[[:space:]]*\(v[0-9]+\.[0-9]+\.[0-9]+\)' \
        "$ROOT/README.md" 2>/dev/null \
      | grep -Eo 'v[0-9]+\.[0-9]+\.[0-9]+' \
      | head -n1 || true
    )"
  fi

  if [[ -z "$BASE_VERSION" ]]; then
    BASE_VERSION="$(
      grep -Eio 'vLLM[^0-9]{0,20}v[0-9]+\.[0-9]+\.[0-9]+' \
        "$ROOT/README.md" 2>/dev/null \
      | grep -Eo 'v[0-9]+\.[0-9]+\.[0-9]+' \
      | head -n1 || true
    )"
  fi

  if [[ -z "$BASE_VERSION" ]]; then
    warn "Could not infer the base vLLM version from README.md."
    if [[ "$MODE" == "precompiled" ]]; then
      die "Use --base-version vX.Y.Z, --commit SHA40, or --wheel PATH."
    fi
  else
    [[ "$BASE_VERSION" == v* ]] || BASE_VERSION="v$BASE_VERSION"
    echo "Detected base vLLM version: $BASE_VERSION"
  fi
}

capture_existing_vllm() {
  section "4/8 Inspect existing vLLM installation"

  local info_file="$LOG_DIR/existing_vllm_${STAMP}.txt"
  if (
    cd /tmp
    env -u PYTHONPATH PYTHONNOUSERSITE=1 "$PYTHON_BIN" - <<'PY'
import pathlib
try:
    import vllm
    print(getattr(vllm, "__version__", "unknown"))
    print(pathlib.Path(vllm.__file__).resolve())
    try:
        import vllm._C
        print(pathlib.Path(vllm._C.__file__).resolve())
    except Exception as exc:
        print(f"NO_BINARY:{type(exc).__name__}:{exc}")
except Exception as exc:
    raise SystemExit(f"{type(exc).__name__}:{exc}")
PY
  ) > "$info_file" 2>&1; then
    cat "$info_file"
    EXISTING_VLLM_VERSION="$(sed -n '1p' "$info_file")"
    EXISTING_VLLM_FILE="$(sed -n '2p' "$info_file")"
    EXISTING_VLLM_BINARY="$(sed -n '3p' "$info_file")"
    EXISTING_VLLM_ROOT="$(dirname "$(dirname "$EXISTING_VLLM_FILE")")"

    if [[ -z "$DONOR_ROOT" && "$EXISTING_VLLM_ROOT" != "$ROOT" && "$EXISTING_VLLM_BINARY" != NO_BINARY:* ]]; then
      DONOR_ROOT="$EXISTING_VLLM_ROOT"
      echo "Automatic donor candidate: $DONOR_ROOT"
    fi
  else
    warn "No currently importable vLLM installation was found."
    cat "$info_file" || true
    EXISTING_VLLM_VERSION=""
    EXISTING_VLLM_FILE=""
    EXISTING_VLLM_BINARY=""
    EXISTING_VLLM_ROOT=""
  fi
}

ensure_build_metadata_deps() {
  section "5/8 Build metadata dependencies"

  "$PYTHON_BIN" -m pip install \
    'setuptools>=77.0.3,<81.0.0' \
    'setuptools-scm>=8.0' \
    'setuptools-rust>=1.9.0' \
    'packaging>=24.2' \
    wheel \
    jinja2 \
    ninja \
    'cmake>=3.26.1'

  "$PYTHON_BIN" - <<'PY'
import importlib
mods = [
    "setuptools",
    "setuptools_scm",
    "setuptools_rust",
    "packaging",
    "wheel",
    "jinja2",
    "torch",
]
for name in mods:
    module = importlib.import_module(name)
    print(f"{name:18s} PASS  {getattr(module, '__version__', '')}")
PY
}

resolve_tag_commit() {
  if [[ -n "$VLLM_COMMIT" ]]; then
    [[ "$VLLM_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]] || die "--commit/KVARN_VLLM_COMMIT must be a 40-character SHA."
    VLLM_COMMIT="${VLLM_COMMIT,,}"
    return 0
  fi

  [[ -n "$BASE_VERSION" ]] || return 1
  command -v git >/dev/null 2>&1 || return 1

  local refs
  refs="$(git ls-remote \
    https://github.com/vllm-project/vllm.git \
    "refs/tags/${BASE_VERSION}" \
    "refs/tags/${BASE_VERSION}^{}" 2>/dev/null || true)"

  VLLM_COMMIT="$(
    printf '%s\n' "$refs" \
      | awk '$2 ~ /\^\{\}$/ {print $1; found=1; exit}
             !fallback {fallback=$1}
             END {if (!found && fallback) print fallback}'
  )"

  [[ "$VLLM_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]
}

detect_variant_candidates() {
  local torch_cuda exact mapped
  torch_cuda="$("$PYTHON_BIN" -c 'import torch; print(torch.version.cuda or "")')"
  exact=""
  mapped=""

  if [[ "$torch_cuda" =~ ^([0-9]+)\.([0-9]+) ]]; then
    exact="cu${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    case "${BASH_REMATCH[1]}" in
      12) mapped="cu129" ;;
      13) mapped="cu130" ;;
    esac
  fi

  if [[ -n "$WHEEL_VARIANT" ]]; then
    printf '%s\n' "$WHEEL_VARIANT"
  fi
  [[ -n "$exact" ]] && printf '%s\n' "$exact"
  [[ -n "$mapped" ]] && printf '%s\n' "$mapped"
  printf '%s\n' "__default__"
}

resolve_remote_wheel() {
  resolve_tag_commit || {
    warn "Could not resolve an exact vLLM commit."
    return 1
  }

  echo "Resolved upstream commit: $VLLM_COMMIT"

  local candidates_file="$LOG_DIR/variants_${STAMP}.txt"
  detect_variant_candidates | awk '!seen[$0]++' > "$candidates_file"
  echo "Wheel variants to probe:"
  sed 's/^/  - /' "$candidates_file"

  local result
  if ! result="$(
    "$PYTHON_BIN" - "$VLLM_COMMIT" "$(uname -m)" "$candidates_file" <<'PY'
import json
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request

commit, arch, candidates_path = sys.argv[1:]
variants = pathlib.Path(candidates_path).read_text().splitlines()

for variant in variants:
    variant_dir = "" if variant == "__default__" else f"{variant}/"
    repo = f"https://wheels.vllm.ai/{commit}/{variant_dir}vllm/"
    meta = urllib.parse.urljoin(repo, "metadata.json")
    print(f"[probe] {meta}", file=sys.stderr)
    try:
        req = urllib.request.Request(meta, headers={"User-Agent": "kvarn-oneclick/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            wheels = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"[skip] {type(exc).__name__}: {exc}", file=sys.stderr)
        continue

    for wheel in wheels:
        if wheel.get("package_name") != "vllm":
            continue
        if arch not in wheel.get("platform_tag", ""):
            continue
        path = wheel.get("path")
        if not path:
            continue
        url = urllib.parse.urljoin(repo, path)
        filename = wheel.get("filename") or url.rsplit("/", 1)[-1]
        print("\t".join([variant, url, filename]))
        raise SystemExit(0)

raise SystemExit(1)
PY
  )"; then
    warn "No exact-commit wheel matched architecture $(uname -m)."
    return 1
  fi

  local selected_variant wheel_url wheel_name
  IFS=$'\t' read -r selected_variant wheel_url wheel_name <<< "$result"
  WHEEL_VARIANT="$selected_variant"
  local cache_dir="$ROOT/.wheel-cache"
  mkdir -p "$cache_dir"
  local wheel_path="$cache_dir/$wheel_name"

  echo "Selected variant : $selected_variant"
  echo "Selected wheel   : $wheel_url"

  if [[ ! -s "$wheel_path" ]]; then
    command -v curl >/dev/null 2>&1 || die "curl is required to download the selected wheel."
    curl -fL \
      --retry 4 \
      --retry-delay 3 \
      --connect-timeout 30 \
      -o "$wheel_path.part" \
      "$wheel_url"
    mv "$wheel_path.part" "$wheel_path"
  else
    echo "Using cached wheel: $wheel_path"
  fi

  validate_wheel_archive "$wheel_path"
  RESOLVED_WHEEL="$wheel_path"
  return 0
}

validate_wheel_archive() {
  local wheel="$1"
  "$PYTHON_BIN" - "$wheel" "$(uname -m)" <<'PY'
import pathlib
import sys
import zipfile

wheel = pathlib.Path(sys.argv[1])
arch = sys.argv[2]

if not wheel.is_file() or wheel.stat().st_size == 0:
    raise SystemExit(f"Wheel is missing or empty: {wheel}")

name = wheel.name
known_arches = ("x86_64", "aarch64")
mentioned = [a for a in known_arches if a in name]
if mentioned and arch not in mentioned:
    raise SystemExit(f"Wheel architecture mismatch: machine={arch}, wheel={name}")

try:
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
except zipfile.BadZipFile as exc:
    raise SystemExit(f"Invalid wheel/ZIP archive: {exc}")

core = sorted(
    n for n in names
    if n.startswith("vllm/") and (
        n.endswith("/_C.abi3.so")
        or n == "vllm/_C.abi3.so"
        or n.startswith("vllm/_C.")
    )
)
if not core:
    raise SystemExit("The wheel does not contain the vLLM CUDA core extension.")

print(f"Wheel archive check: PASS ({wheel.name})")
print(f"Core extension: {core[0]}")
PY
}

donor_is_compatible() {
  [[ -n "$DONOR_ROOT" ]] || return 1
  DONOR_ROOT="$(readlink -f "$DONOR_ROOT" 2>/dev/null || printf '%s' "$DONOR_ROOT")"
  [[ -d "$DONOR_ROOT/vllm" ]] || return 1

  local donor_info
  if ! donor_info="$(
    cd /tmp
    env -u PYTHONPATH \
      PYTHONNOUSERSITE=1 \
      PYTHONPATH="$DONOR_ROOT" \
      "$PYTHON_BIN" - <<'PY'
import pathlib
import vllm
print(getattr(vllm, "__version__", "unknown"))
print(pathlib.Path(vllm.__file__).resolve())
import vllm._C
print(pathlib.Path(vllm._C.__file__).resolve())
PY
  )"; then
    warn "Donor vLLM cannot be imported from $DONOR_ROOT."
    return 1
  fi

  DONOR_VERSION="$(sed -n '1p' <<< "$donor_info")"
  DONOR_FILE="$(sed -n '2p' <<< "$donor_info")"
  DONOR_BINARY="$(sed -n '3p' <<< "$donor_info")"

  echo "Donor version : $DONOR_VERSION"
  echo "Donor source  : $DONOR_FILE"
  echo "Donor binary  : $DONOR_BINARY"

  if [[ -n "$BASE_VERSION" ]]; then
    local expected actual
    expected="$(normalize_version "$BASE_VERSION")"
    actual="$(normalize_version "$DONOR_VERSION")"
    if [[ "$actual" != "$expected" ]]; then
      warn "Donor version $actual does not match KVarN base version $expected."
      return 1
    fi
  fi

  [[ "$DONOR_BINARY" == "$DONOR_ROOT"/* ]] || {
    warn "Donor binary is outside donor root."
    return 1
  }

  return 0
}

make_donor_carrier() {
  donor_is_compatible || return 1

  local carrier_dir="$ROOT/.wheel-cache"
  mkdir -p "$carrier_dir"
  local carrier="$carrier_dir/vllm-donor-${DONOR_VERSION//[^A-Za-z0-9._-]/_}-$(uname -m).whl"

  "$PYTHON_BIN" - "$DONOR_ROOT" "$carrier" <<'PY'
import pathlib
import sys
import zipfile

root = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(sys.argv[2]).resolve()
pkg = root / "vllm"

include = []
for path in pkg.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(root).as_posix()
    name = path.name

    keep = False
    if name == "vllm-rs":
        keep = True
    elif ".so" in name:
        keep = True
    elif rel.startswith("vllm/vllm_flash_attn/") and path.suffix == ".py":
        keep = True
    elif rel.startswith("vllm/third_party/triton_kernels/"):
        keep = True
    elif rel.startswith("vllm/third_party/flashmla/"):
        keep = True
    elif rel.startswith("vllm/third_party/deep_gemm/"):
        keep = True

    if keep:
        include.append((path, rel))

core = [
    rel for _, rel in include
    if rel == "vllm/_C.abi3.so" or rel.startswith("vllm/_C.")
]
if not core:
    raise SystemExit(f"No vLLM core binary was found under {pkg}")

out.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
    for path, rel in include:
        zf.write(path, rel)

print(f"Created donor binary carrier: {out}")
print(f"Files included: {len(include)}")
print(f"Core: {core[0]}")
PY

  validate_wheel_archive "$carrier"
  RESOLVED_WHEEL="$carrier"
  return 0
}

clean_registration() {
  info "Removing the current vLLM package registration before installation."
  "$PYTHON_BIN" -m pip uninstall -y vllm || true

  # Remove only top-level native artifacts that the precompiled installer replaces.
  # Source-controlled Python files and experiment files are not touched.
  find "$ROOT/vllm" -maxdepth 1 -type f \
    \( -name '_C*.so' \
       -o -name '_moe_C*.so' \
       -o -name '_flashmla_C*.so' \
       -o -name '_flashmla_extension_C*.so' \
       -o -name '_sparse_flashmla_C*.so' \
       -o -name 'cumem_allocator*.so' \
       -o -name 'spinloop*.so' \
       -o -name 'vllm-rs' \) \
    -print -delete 2>/dev/null || true

  # Nested native files are generated/extracted artifacts, not Python source.
  find "$ROOT/vllm/vllm_flash_attn" \
       "$ROOT/vllm/third_party/deep_gemm" \
       "$ROOT/vllm/third_party/flashmla" \
       -type f -name '*.so*' -print -delete 2>/dev/null || true
}

install_with_wheel() {
  local wheel="$1"
  local label="$2"
  local rc

  section "6/8 Install KVarN using $label"
  clean_registration

  set +e
  (
    cd "$ROOT"
    env \
      -u PYTHONPATH \
      PYTHONNOUSERSITE=1 \
      VLLM_USE_PRECOMPILED=1 \
      VLLM_PRECOMPILED_WHEEL_LOCATION="$wheel" \
      "$PYTHON_BIN" -m pip install -e . \
        --no-build-isolation \
        --no-deps
  )
  rc=$?
  set -e

  if (( rc != 0 )); then
    warn "Installation strategy '$label' failed with exit code $rc."
    return "$rc"
  fi

  INSTALL_STRATEGY="$label"
  return 0
}

can_source_build() {
  command -v nvcc >/dev/null 2>&1 || {
    warn "nvcc not found; local CUDA source compilation is unavailable."
    return 1
  }
  command -v gcc >/dev/null 2>&1 || {
    warn "gcc not found; local source compilation is unavailable."
    return 1
  }
  command -v g++ >/dev/null 2>&1 || {
    warn "g++ not found; local source compilation is unavailable."
    return 1
  }
  command -v cmake >/dev/null 2>&1 || return 1
  command -v ninja >/dev/null 2>&1 || return 1
  return 0
}

install_from_source() {
  local rc
  section "6/8 Install KVarN by local source compilation"

  can_source_build || {
    warn "Source compilation prerequisites are incomplete."
    return 1
  }
  echo "nvcc  : $(nvcc --version | tail -n1)"
  echo "gcc   : $(gcc --version | head -n1)"
  echo "g++   : $(g++ --version | head -n1)"
  echo "cmake : $(cmake --version | head -n1)"
  echo "ninja : $(ninja --version)"

  clean_registration

  set +e
  (
    cd "$ROOT"
    env \
      -u PYTHONPATH \
      -u VLLM_USE_PRECOMPILED \
      -u VLLM_PRECOMPILED_WHEEL_LOCATION \
      -u VLLM_PRECOMPILED_WHEEL_COMMIT \
      -u VLLM_PRECOMPILED_WHEEL_VARIANT \
      PYTHONNOUSERSITE=1 \
      MAX_JOBS="$MAX_JOBS_VALUE" \
      "$PYTHON_BIN" -m pip install -e . \
        --no-build-isolation \
        --no-deps
  )
  rc=$?
  set -e

  if (( rc != 0 )); then
    warn "Local source compilation failed with exit code $rc."
    return "$rc"
  fi

  INSTALL_STRATEGY="local-source-build"
  return 0
}

static_check() {
  section "7/8 Static and import verification"

  if (
    cd /tmp
    env -u PYTHONPATH PYTHONNOUSERSITE=1 "$PYTHON_BIN" - "$ROOT" <<'PY'
import importlib
import json
import pathlib
import pkgutil
import sys
from typing import get_args

expected_root = pathlib.Path(sys.argv[1]).resolve()
result = {
    "python": sys.executable,
    "expected_root": str(expected_root),
}

import torch
result["torch"] = torch.__version__
result["torch_cuda"] = torch.version.cuda
result["cuda_available"] = torch.cuda.is_available()

import vllm
vllm_file = pathlib.Path(vllm.__file__).resolve()
result["vllm_version"] = getattr(vllm, "__version__", "unknown")
result["vllm_file"] = str(vllm_file)

try:
    vllm_file.relative_to(expected_root)
except ValueError as exc:
    raise SystemExit(
        "Wrong vLLM source is loaded.\n"
        f"Expected under: {expected_root}\n"
        f"Actually loaded: {vllm_file}"
    ) from exc

import vllm.config.cache as cache_config
cache_file = pathlib.Path(cache_config.__file__).resolve()
result["cache_config"] = str(cache_file)

allowed = set(get_args(cache_config.CacheDType))
if "kvarn_k4v2_g128" not in allowed:
    raise SystemExit(
        "CacheDType does not contain kvarn_k4v2_g128.\n"
        f"Cache config: {cache_file}\n"
        f"Allowed: {sorted(map(str, allowed))}"
    )
result["kvarn_dtype"] = "PASS"

import vllm._C
binary_file = pathlib.Path(vllm._C.__file__).resolve()
result["vllm_binary"] = str(binary_file)
if not binary_file.is_file():
    raise SystemExit(f"vLLM native extension is missing: {binary_file}")

backend_pkg = importlib.import_module("vllm.v1.attention.backends")
backend_names = [
    mod.name
    for mod in pkgutil.iter_modules(backend_pkg.__path__)
    if "kvarn" in mod.name.lower()
]
if not backend_names:
    raise SystemExit(
        "No importable KVarN backend module was found in "
        "vllm.v1.attention.backends."
    )

imported = []
errors = {}
for name in backend_names:
    fqname = f"vllm.v1.attention.backends.{name}"
    try:
        importlib.import_module(fqname)
        imported.append(fqname)
    except Exception as exc:
        errors[fqname] = f"{type(exc).__name__}: {exc}"

if not imported:
    raise SystemExit(f"KVarN backend import failed: {errors}")

result["kvarn_backends"] = imported
result["status"] = "PASS"
print(json.dumps(result, indent=2, ensure_ascii=False))
PY
  ); then
    STATIC_STATUS="PASS"
    echo
    echo "STATIC_CHECK=PASS"
    return 0
  else
    STATIC_STATUS="FAIL"
    echo
    echo "STATIC_CHECK=FAIL"
    return 1
  fi
}

runtime_smoke() {
  section "8/8 KVarN runtime smoke test"

  if [[ "$RUN_SMOKE" == "no" ]]; then
    RUNTIME_STATUS="SKIPPED_BY_USER"
    echo "Runtime smoke test skipped."
    return 0
  fi

  if [[ ! -e "$SMOKE_MODEL" ]]; then
    if [[ "$RUN_SMOKE" == "required" ]]; then
      RUNTIME_STATUS="FAIL_MODEL_MISSING"
      die "Required smoke model does not exist: $SMOKE_MODEL"
    fi
    RUNTIME_STATUS="SKIPPED_MODEL_MISSING"
    warn "Smoke model not found; static verification passed, runtime smoke skipped: $SMOKE_MODEL"
    return 0
  fi

  local smoke_cmd=(
    env -u PYTHONPATH
    PYTHONNOUSERSITE=1
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    "$PYTHON_BIN" - "$SMOKE_MODEL"
  )

  local runner=("${smoke_cmd[@]}")
  if command -v timeout >/dev/null 2>&1; then
    runner=(timeout 300s "${smoke_cmd[@]}")
  fi

  if (
    cd /tmp
    "${runner[@]}" <<'PY'
import gc
import json
import sys
import torch

from vllm import LLM, SamplingParams

model = sys.argv[1]
print(f"Loading runtime smoke model: {model}")

llm = LLM(
    model=model,
    tokenizer=model,
    trust_remote_code=True,
    dtype="float16",
    kv_cache_dtype="kvarn_k4v2_g128",
    block_size=128,
    max_model_len=2048,
    gpu_memory_utilization=0.80,
    max_num_seqs=1,
    enable_prefix_caching=False,
    disable_log_stats=True,
    enforce_eager=True,
    seed=2026,
)

outputs = llm.generate(
    ["Return one word: OK"],
    SamplingParams(temperature=0.0, max_tokens=1),
)
token_ids = outputs[0].outputs[0].token_ids
text = outputs[0].outputs[0].text

print(json.dumps({
    "generated_token_ids": token_ids,
    "generated_text": text,
    "status": "PASS",
}, ensure_ascii=False, indent=2))

del outputs
del llm
gc.collect()
torch.cuda.empty_cache()

try:
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
except Exception:
    pass

try:
    from vllm.distributed.parallel_state import destroy_distributed_environment
    destroy_distributed_environment()
except Exception:
    pass

print("RUNTIME_SMOKE=PASS")
PY
  ); then
    RUNTIME_STATUS="PASS"
    return 0
  fi

  RUNTIME_STATUS="FAIL"
  if [[ "$RUN_SMOKE" == "required" ]]; then
    return 1
  fi
  warn "Runtime smoke failed. Static verification may still be valid; inspect the log."
  return 1
}

attempt_static_after_install() {
  if static_check; then
    return 0
  fi
  warn "The selected installation strategy completed, but static verification failed."
  return 1
}

perform_install() {
  confirm_replacement
  ensure_build_metadata_deps

  RESOLVED_WHEEL=""

  if [[ -n "$EXPLICIT_WHEEL" ]]; then
    if [[ -f "$EXPLICIT_WHEEL" ]]; then
      validate_wheel_archive "$EXPLICIT_WHEEL"
    elif [[ ! "$EXPLICIT_WHEEL" =~ ^https?:// ]]; then
      die "Explicit wheel is neither a local file nor an HTTP(S) URL: $EXPLICIT_WHEEL"
    fi
    if install_with_wheel "$EXPLICIT_WHEEL" "explicit-wheel"; then
      attempt_static_after_install && return 0
    fi
    [[ "$MODE" == "auto" ]] || die "Explicit wheel installation failed verification."
    warn "Continuing with automatic fallback strategies."
  fi

  if [[ "$MODE" == "auto" || "$MODE" == "precompiled" ]]; then
    RESOLVED_WHEEL=""
    if resolve_remote_wheel; then
      if install_with_wheel "$RESOLVED_WHEEL" "exact-commit-wheel"; then
        if attempt_static_after_install; then
          return 0
        fi
        warn "Exact-commit wheel did not pass verification."
      fi
    fi
    [[ "$MODE" == "precompiled" ]] && die "No verified precompiled installation strategy succeeded."
  fi

  if [[ "$MODE" == "auto" || "$MODE" == "donor" ]]; then
    RESOLVED_WHEEL=""
    if make_donor_carrier; then
      if install_with_wheel "$RESOLVED_WHEEL" "existing-vllm-binary-carrier"; then
        if attempt_static_after_install; then
          return 0
        fi
        warn "Donor binary carrier did not pass verification."
      fi
    fi
    [[ "$MODE" == "donor" ]] && die "No compatible donor vLLM installation succeeded."
  fi

  if [[ "$MODE" == "auto" || "$MODE" == "source" ]]; then
    if can_source_build; then
      if install_from_source; then
        attempt_static_after_install && return 0
      fi
    fi
  fi

  die "All permitted installation strategies failed. Review: $LOG_FILE"
}

main() {
  preflight
  inspect_torch
  detect_base_version
  capture_existing_vllm

  if (( CHECK_ONLY )); then
    if static_check; then
      if ! runtime_smoke; then
        write_report
        die "Current installation passed imports but failed the runtime smoke test."
      fi
      write_report
      section "Result"
      echo "Static verification : $STATIC_STATUS"
      echo "Runtime smoke       : $RUNTIME_STATUS"
      echo "Report              : $REPORT_FILE"
      echo "Log                 : $LOG_FILE"
      [[ "$STATIC_STATUS" == "PASS" ]] || exit 1
      [[ "$RUNTIME_STATUS" != "FAIL" ]] || exit 1
      exit 0
    fi
    write_report
    die "Current installation failed static verification."
  fi

  perform_install

  # perform_install already runs static_check after each strategy.
  if ! runtime_smoke; then
    write_report
    die "Installation passed imports but failed the KVarN runtime smoke test."
  fi

  write_report
  section "Result"
  echo "Install strategy    : $INSTALL_STRATEGY"
  echo "Static verification : $STATIC_STATUS"
  echo "Runtime smoke       : $RUNTIME_STATUS"
  echo "Report              : $REPORT_FILE"
  echo "Log                 : $LOG_FILE"

  [[ "$STATIC_STATUS" == "PASS" ]] || exit 1
  [[ "$RUN_SMOKE" != "required" || "$RUNTIME_STATUS" == "PASS" ]] || exit 1

  echo
  echo "KVarN installation and verification completed successfully."
}

main "$@"
