"""Markdown report generation for fNIRS experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_experiment_report(
    *,
    experiment: dict[str, Any],
    result: dict[str, Any] | None,
    explanation: dict[str, Any] | None,
    output_dir: str | Path,
) -> Path:
    output_path = Path(output_dir) / "report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = (result or {}).get("metrics", {})
    folds = (result or {}).get("folds", [])
    top_channels = (explanation or {}).get("top_channels", [])

    lines = [
        f"# {experiment.get('name', 'fNIRS Experiment')} 报告",
        "",
        "## 实验概览",
        f"- 实验 ID：`{experiment.get('id', experiment.get('experiment_id', 'unknown'))}`",
        f"- 数据集：`{experiment.get('dataset_id') or experiment.get('dataset_path') or 'demo'}`",
        f"- 验证策略：`{experiment.get('validation_strategy', 'loso')}`",
        f"- 模型：`{(experiment.get('model') or {}).get('model_family', 'cnn-lstm')}`",
        "",
        "## 关键指标",
        f"- Accuracy：{metrics.get('accuracy', 'N/A')}",
        f"- 样本数：{metrics.get('n_samples', 'N/A')}",
        "",
        "## Fold 结果",
    ]
    if folds:
        for fold in folds:
            lines.append(
                f"- {fold.get('fold_name')}: accuracy={fold.get('accuracy')}, "
                f"train={fold.get('train_size')}, val={fold.get('val_size')}"
            )
    else:
        lines.append("- 暂无 fold 结果。")
    lines.extend(["", "## 可解释性摘要"])
    if top_channels:
        for item in top_channels:
            lines.append(f"- {item.get('channel')}: {item.get('importance')}")
    else:
        lines.append("- 暂无解释结果。")
    lines.extend(
        [
            "",
            "## 配置快照",
            "```json",
            json.dumps(experiment, ensure_ascii=False, indent=2),
            "```",
            "",
            "## 边界说明",
            "本地 v1 优先保证端到端流程、subject-wise 验证与结果追踪。正式科研结论仍需结合更大样本、预注册方案和人工复核。",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

