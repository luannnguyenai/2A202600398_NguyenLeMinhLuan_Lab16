"""
run_benchmark.py
-----------------
Script chính để chạy benchmark ReAct vs Reflexion.

Cách dùng:
    # Mock mode (nhanh, test scaffold)
    python run_benchmark.py --dataset data/hotpot_mini.json --out-dir outputs/sample_run --mode mock

    # Real mode (Ollama + HotpotQA)
    python run_benchmark.py --dataset data/hotpot_real_120.json --out-dir outputs/real_run --mode real --limit 100
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.schemas import RunRecord
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)
console = Console()


def _run_agent(agent, examples, desc: str) -> list[RunRecord]:
    """Chạy agent trên list examples với tqdm progress bar."""
    records: list[RunRecord] = []
    for example in tqdm(examples, desc=desc, unit="q"):
        records.append(agent.run(example))
    return records


@app.command()
def main(
    dataset: str = typer.Option("data/hotpot_mini.json", help="Path tới file dataset JSON"),
    out_dir: str = typer.Option("outputs/sample_run", help="Thư mục lưu kết quả"),
    reflexion_attempts: int = typer.Option(3, help="Số lần retry tối đa của ReflexionAgent"),
    mode: str = typer.Option("real", help="'real' dùng Ollama, 'mock' dùng mock_runtime"),
    model: str = typer.Option("llama3.1:8b", help="Ollama model tag"),
    limit: Optional[int] = typer.Option(100, help="Số example tối đa (None = toàn bộ)"),
) -> None:
    """Chạy benchmark ReAct vs Reflexion và lưu report.json + report.md."""

    if mode not in ("real", "mock"):
        raise typer.BadParameter("--mode phải là 'real' hoặc 'mock'")

    # Load dataset
    all_examples = load_dataset(dataset)
    if limit is not None:
        examples = all_examples[:limit]
    else:
        examples = all_examples
    console.log(f"[cyan]Dataset:[/cyan] {dataset} — {len(examples)} examples (mode={mode})")

    # Khởi tạo agents
    if mode == "real":
        from src.reflexion_lab.llm_runtime import OllamaClient

        client = OllamaClient(model=model)
        console.log(f"[cyan]Ollama model:[/cyan] {model}")
        react = ReActAgent(client=client, mode="real")
        reflexion = ReflexionAgent(max_attempts=reflexion_attempts, client=client, mode="real")
    else:
        react = ReActAgent(mode="mock")
        reflexion = ReflexionAgent(max_attempts=reflexion_attempts, mode="mock")

    # Chạy benchmark
    console.rule("[bold blue]ReAct Agent")
    react_records = _run_agent(react, examples, desc="ReAct")

    console.rule("[bold magenta]Reflexion Agent")
    reflexion_records = _run_agent(reflexion, examples, desc="Reflexion")

    all_records = react_records + reflexion_records

    # Lưu JSONL
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)

    # Lưu dataset snapshot
    snapshot_path = out_path / "dataset_snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps([e.model_dump() for e in examples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Build và lưu report
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)

    # In bảng tóm tắt
    table = Table(title="Benchmark Summary", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("ReAct", justify="right")
    table.add_column("Reflexion", justify="right")
    table.add_column("Δ (Reflex − React)", justify="right", style="cyan")

    react_s = report.summary.get("react", {})
    reflex_s = report.summary.get("reflexion", {})
    delta = report.summary.get("delta_reflexion_minus_react", {})

    table.add_row("EM", str(react_s.get("em", 0)), str(reflex_s.get("em", 0)), f"{delta.get('em_abs', 0):+.4f}")
    table.add_row(
        "Avg attempts",
        str(react_s.get("avg_attempts", 0)),
        str(reflex_s.get("avg_attempts", 0)),
        f"{delta.get('attempts_abs', 0):+.4f}",
    )
    table.add_row(
        "Avg tokens",
        str(react_s.get("avg_token_estimate", 0)),
        str(reflex_s.get("avg_token_estimate", 0)),
        f"{delta.get('tokens_abs', 0):+.2f}",
    )
    table.add_row(
        "Avg latency (ms)",
        str(react_s.get("avg_latency_ms", 0)),
        str(reflex_s.get("avg_latency_ms", 0)),
        f"{delta.get('latency_abs', 0):+.2f}",
    )

    console.print(table)
    console.print(f"\n[green]✓ report.json:[/green] {json_path}")
    console.print(f"[green]✓ report.md:[/green] {md_path}")
    console.print(f"[green]✓ dataset_snapshot.json:[/green] {snapshot_path}")
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
