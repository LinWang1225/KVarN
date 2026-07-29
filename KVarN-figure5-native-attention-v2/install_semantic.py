#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the Figure-5 native-attention benchmark without line-number patches.

This installer is semantic and idempotent:
1. Copies the new benchmark/probe files into a KVarN checkout.
2. Inserts the dormant Figure-5 hook immediately before the backend output write.
3. Adds ``import os`` only when the target file does not already import it.
4. Refuses ambiguous edits instead of modifying an unexpected code layout.
"""

from __future__ import annotations

import argparse
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path


KVAR_HOOK = '''\
        # Benchmark-only Figure 5 probe. Normal serving never imports or calls
        # the probe because the environment variable is absent.
        if os.environ.get("VLLM_FIGURE5_PROBE_OUTPUT"):
            from vllm.v1.attention.figure5_probe import probe_attention_output

            attn_out = probe_attention_output(
                method="kvarn",
                impl=self,
                layer=layer,
                query=q,
                key=key[:N].view(N, self.num_kv_heads, self.head_size),
                value=value[:N].view(N, self.num_kv_heads, self.head_size),
                attn_metadata=attn_metadata,
                native_output=attn_out,
            )
'''

TQ_HOOK = '''\
        # Benchmark-only Figure 5 probe. Normal serving never imports or calls
        # the probe because the environment variable is absent.
        if os.environ.get("VLLM_FIGURE5_PROBE_OUTPUT"):
            from vllm.v1.attention.figure5_probe import probe_attention_output

            attn_out = probe_attention_output(
                method="turboquant",
                impl=self,
                layer=layer,
                query=q,
                key=key[:N].view(N, self.num_kv_heads, self.head_size),
                value=value[:N].view(N, self.num_kv_heads, self.head_size),
                attn_metadata=attn_metadata,
                native_output=attn_out,
            )
'''

KVAR_ANCHOR = '''\
        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
'''

TQ_ANCHOR = '''\
        # Write into output buffer: attn_out is (N, Hq, D)
        # output may be 2D (N, Hq*D) or 3D (N, Hq, D)
        if output.ndim == 3:
'''


def add_import_os(text: str, path: Path) -> str:
    if "\nimport os\n" in text or text.startswith("import os\n"):
        return text

    for anchor in ("import math\n", "import functools\n"):
        if text.count(anchor) == 1:
            return text.replace(anchor, anchor + "import os\n", 1)

    raise RuntimeError(
        f"{path}: could not find a unique import anchor for `import os`."
    )


def insert_hook(path: Path, *, hook: str, anchor: str) -> bool:
    text = path.read_text(encoding="utf-8")

    if "VLLM_FIGURE5_PROBE_OUTPUT" in text:
        print(f"[skip] hook already present: {path}")
        return False

    text = add_import_os(text, path)

    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected exactly one semantic anchor, found {count}. "
            "The backend layout is different; inspect its forward() method."
        )

    path.write_text(text.replace(anchor, hook + anchor, 1), encoding="utf-8")
    print(f"[edit] inserted Figure-5 hook: {path}")
    return True


def copy_new_files(package_root: Path, repo: Path, *, force: bool) -> list[Path]:
    source_root = package_root / "files"
    if not source_root.is_dir():
        raise RuntimeError(f"Missing package files directory: {source_root}")

    copied: list[Path] = []
    for source in sorted(source_root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists():
            if source.read_bytes() == target.read_bytes():
                print(f"[skip] identical file: {relative}")
                continue
            if not force:
                raise RuntimeError(
                    f"Target already exists with different content: {relative}\n"
                    "Review it or rerun with --force-new-files."
                )

        shutil.copy2(source, target)
        copied.append(target)
        print(f"[copy] {relative}")

    return copied


def git_root(path: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        check=True,
        text=True,
        capture_output=True,
    )
    return Path(result.stdout.strip()).resolve()


def validate(repo: Path, files: list[Path]) -> None:
    python_files = [path for path in files if path.suffix == ".py"]
    python_files += [
        repo / "vllm/v1/attention/backends/kvarn_attn.py",
        repo / "vllm/v1/attention/backends/turboquant_attn.py",
    ]

    for path in python_files:
        py_compile.compile(str(path), doraise=True)

    subprocess.run(["git", "-C", str(repo), "diff", "--check"], check=True)
    print("[ok] Python syntax and `git diff --check` passed.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="KVarN checkout; defaults to the current directory.",
    )
    parser.add_argument(
        "--force-new-files",
        action="store_true",
        help="Overwrite benchmark/probe files that already exist with different content.",
    )
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parent
    repo = git_root(args.repo.resolve())

    required = [
        repo / "vllm/v1/attention/backends/kvarn_attn.py",
        repo / "vllm/v1/attention/backends/turboquant_attn.py",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "This does not look like the expected KVarN checkout. Missing:\n"
            + "\n".join(str(path) for path in missing)
        )

    copied = copy_new_files(package_root, repo, force=args.force_new_files)

    edited: list[Path] = []
    if insert_hook(required[0], hook=KVAR_HOOK, anchor=KVAR_ANCHOR):
        edited.append(required[0])
    if insert_hook(required[1], hook=TQ_HOOK, anchor=TQ_ANCHOR):
        edited.append(required[1])

    validate(repo, copied + edited)

    print("\nInstalled. Review changes with:")
    print(f"  git -C {repo} diff --stat")
    print(f"  git -C {repo} diff -- vllm/v1/attention/backends")
    print("\nThen reinstall the editable package:")
    print("  VLLM_USE_PRECOMPILED=1 pip install -e .")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
