#!/usr/bin/env python3
"""HumanEval-specific helpers for trajectory-divergence experiments.

The generated code is untrusted. ``evaluate_humaneval_candidate`` applies basic
process/resource isolation, but it is not a hardened security sandbox. Run the
experiment in an isolated container/VM with no secrets and no sensitive mounts.
"""

from __future__ import annotations

import hashlib
import os
import re
import resource
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Iterable

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_subsequence(sequence: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(sequence):
        return None
    limit = len(sequence) - len(needle) + 1
    for index in range(limit):
        if sequence[index : index + len(needle)] == needle:
            return index
    return None


def split_thinking_tokens(tokenizer: Any, output_token_ids: Iterable[int]) -> dict[str, Any]:
    """Split generated tokens at the first ``</think>`` marker when present."""
    output_ids = [int(value) for value in output_token_ids]
    marker_ids = [
        int(value)
        for value in tokenizer.encode("</think>", add_special_tokens=False)
    ]
    marker_start = find_subsequence(output_ids, marker_ids)
    if marker_start is None:
        return {
            "thinking_boundary_detected": False,
            "thinking_end_step": None,
            "thinking_tokens": None,
            "final_tokens": None,
        }
    thinking_tokens = marker_start + len(marker_ids)
    return {
        "thinking_boundary_detected": True,
        "thinking_end_step": thinking_tokens,
        "thinking_tokens": thinking_tokens,
        "final_tokens": max(0, len(output_ids) - thinking_tokens),
    }


def visible_after_thinking(text: str) -> str:
    """Return text after the last closing think marker, or the full text."""
    marker = "</think>"
    if marker in text:
        return text.rsplit(marker, 1)[1].strip()
    return text.strip()


def extract_python_candidate(output_text: str, entry_point: str) -> dict[str, Any]:
    """Extract the final Python candidate from a reasoning-model response.

    Preference order:
      1. last fenced block containing ``def <entry_point>``;
      2. last fenced code block;
      3. all visible text after ``</think>``.
    """
    visible = visible_after_thinking(output_text)
    blocks = [block.strip() for block in _CODE_FENCE_RE.findall(visible) if block.strip()]
    entry_pattern = re.compile(rf"\bdef\s+{re.escape(entry_point)}\s*\(")

    selected: str
    source: str
    matching = [block for block in blocks if entry_pattern.search(block)]
    if matching:
        selected = matching[-1]
        source = "fenced_entry_point"
    elif blocks:
        selected = blocks[-1]
        source = "fenced_last"
    else:
        selected = visible
        source = "visible_text"

    # Remove a few common non-code lead-ins when no fence was used.
    selected = re.sub(
        r"^(?:Here(?:'s| is)\s+(?:the\s+)?(?:implementation|code)\s*:?\s*)",
        "",
        selected.strip(),
        flags=re.IGNORECASE,
    )
    return {
        "visible_output": visible,
        "candidate_code": selected.strip(),
        "candidate_source": source,
        "candidate_code_sha256": sha256_text(selected.strip()),
        "contains_entry_point_definition": bool(entry_pattern.search(selected)),
    }


def build_humaneval_program(
    *,
    prompt: str,
    candidate_code: str,
    test: str,
    entry_point: str,
) -> tuple[str, str]:
    """Build an executable HumanEval test program.

    HumanEval originally expects a completion appended to ``prompt``. Chat
    models often return a complete function instead. Support both forms.
    """
    entry_pattern = re.compile(rf"\bdef\s+{re.escape(entry_point)}\s*\(")
    if entry_pattern.search(candidate_code):
        prompt_match = entry_pattern.search(prompt)
        prelude = prompt[: prompt_match.start()] if prompt_match is not None else ""
        solution = prelude.rstrip() + ("\n\n" if prelude.strip() else "") + candidate_code.rstrip() + "\n"
        assembly = "standalone_function_with_prompt_prelude" if prelude.strip() else "standalone_function"
    else:
        completion = candidate_code.rstrip()
        first_nonempty = next((line for line in completion.splitlines() if line.strip()), "")
        if first_nonempty and not first_nonempty[0].isspace():
            completion = textwrap.indent(completion, "    ")
        solution = prompt.rstrip() + "\n" + completion + "\n"
        assembly = "prompt_plus_completion"

    program = (
        solution
        + "\n"
        + test.rstrip()
        + "\n\n"
        + f"check({entry_point})\n"
    )
    return program, assembly


def _limit_child_resources(cpu_seconds: int, memory_mb: int) -> None:
    """Best-effort POSIX resource guard for the evaluator child process."""
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        memory_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, OSError):
        # Some limits are unavailable or restricted in containers. The parent
        # wall-clock timeout still applies.
        pass


def classify_execution(returncode: int | None, stderr: str, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "passed"
    lowered = stderr.lower()
    if "syntaxerror" in lowered or "indentationerror" in lowered or "taberror" in lowered:
        return "syntax_error"
    if "memoryerror" in lowered:
        return "memory_error"
    return "failed"


def evaluate_humaneval_candidate(
    *,
    prompt: str,
    candidate_code: str,
    test: str,
    entry_point: str,
    timeout_seconds: float = 30.0,
    memory_mb: int = 1024,
) -> dict[str, Any]:
    """Execute one HumanEval candidate in a short-lived isolated process."""
    program, assembly = build_humaneval_program(
        prompt=prompt,
        candidate_code=candidate_code,
        test=test,
        entry_point=entry_point,
    )
    started = time.perf_counter()
    timed_out = False
    returncode: int | None = None
    stdout = ""
    stderr = ""

    with tempfile.TemporaryDirectory(prefix="kvarn_humaneval_") as temp_dir:
        script_path = Path(temp_dir) / "candidate_test.py"
        script_path.write_text(program, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
        cpu_seconds = max(1, int(timeout_seconds) + 1)
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=temp_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=(
                    lambda: _limit_child_resources(cpu_seconds=cpu_seconds, memory_mb=memory_mb)
                )
                if os.name == "posix"
                else None,
            )
            returncode = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""

    elapsed = time.perf_counter() - started
    result_class = classify_execution(returncode, stderr, timed_out)
    return {
        "passed": result_class == "passed",
        "result": result_class,
        "returncode": returncode,
        "execution_seconds": elapsed,
        "assembly": assembly,
        "program_sha256": sha256_text(program),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "timed_out": timed_out,
    }
