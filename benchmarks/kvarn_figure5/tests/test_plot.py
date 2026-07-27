from pathlib import Path

import pandas as pd

from plot_figure5 import plot_figure


def test_plot(tmp_path: Path):
    rows = []
    for method, label in [("kvarn", "KVarN K4/V2 G128"), ("turboquant", "TurboQuant 3-bit NC")]:
        for context in (1024, 2048):
            rows.append(
                {
                    "method": method,
                    "label": label,
                    "context_tokens": context,
                    "mean_abs_logprob_error": 0.1,
                    "top1_agreement": 0.9,
                    "prompt_tokens_per_second": 1000.0,
                    "fp16_prompt_tokens_per_second": 900.0,
                }
            )
    source = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    output = tmp_path / "figure"
    plot_figure(source, output)
    assert output.with_suffix(".png").exists()
    assert output.with_suffix(".pdf").exists()
    assert output.with_suffix(".svg").exists()
