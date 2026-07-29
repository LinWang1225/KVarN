# Fix for `patch does not apply`

The monolithic patch contains line-context hunks for two rapidly changing files:

- `vllm/v1/attention/backends/kvarn_attn.py`
- `vllm/v1/attention/backends/turboquant_attn.py`

If the checkout has a newer commit or local edits, `git apply --check` can reject
those hunks even though the intended insertion point still exists.

Use the semantic installer instead of `git apply`:

```bash
cd /data/wanglin/KVarN

git status --short
git diff > /tmp/kvarn-before-figure5.patch
git diff --cached > /tmp/kvarn-before-figure5-staged.patch

python /path/to/KVarN-figure5-native-attention-v2/install_semantic.py \
  --repo /data/wanglin/KVarN
```

If benchmark files from an earlier attempt already exist with different content,
review them first. To intentionally replace only those package files:

```bash
python /path/to/KVarN-figure5-native-attention-v2/install_semantic.py \
  --repo /data/wanglin/KVarN \
  --force-new-files
```

Then inspect and reinstall:

```bash
git diff --check
git diff --stat
git diff -- vllm/v1/attention/backends/kvarn_attn.py
git diff -- vllm/v1/attention/backends/turboquant_attn.py

VLLM_USE_PRECOMPILED=1 pip install -e .
```

The installer is idempotent: if the hook already exists, it skips it. It aborts
rather than editing when the expected output-write anchor is missing or appears
more than once.
