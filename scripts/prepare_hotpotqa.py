"""
scripts/prepare_hotpotqa.py
----------------------------
Tải và chuyển đổi HotpotQA dev-distractor sang định dạng QAExample.

CLI:
    python scripts/prepare_hotpotqa.py --n 120 --seed 42 --out data/hotpot_real_120.json
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

import requests
import typer
from rich.console import Console

console = Console()
app = typer.Typer(add_completion=False)

# URL raw của HotpotQA dev-distractor (~47MB)
HOTPOT_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
DEFAULT_CACHE = Path("data/hotpot_dev_distractor_v1.json")


def download_hotpotqa(cache_path: Path) -> list[dict]:
    """Tải file JSON HotpotQA nếu chưa có trong cache, trả về list raw items."""
    if cache_path.exists():
        console.log(f"[green]Dùng cache:[/green] {cache_path}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    console.log(f"[yellow]Đang tải HotpotQA từ:[/yellow] {HOTPOT_URL}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(HOTPOT_URL, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunks: list[bytes] = []
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            chunks.append(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                console.print(f"  {pct:.1f}%", end="\r")
        raw_bytes = b"".join(chunks)

    cache_path.write_bytes(raw_bytes)
    console.log(f"[green]Đã lưu cache:[/green] {cache_path} ({len(raw_bytes) / 1e6:.1f} MB)")
    return json.loads(raw_bytes.decode("utf-8"))


def _level_to_difficulty(level: str) -> str:
    """Map HotpotQA level string sang difficulty literal."""
    mapping = {"easy": "easy", "medium": "medium", "hard": "hard"}
    return mapping.get(level, "medium")


def convert_item(item: dict, rng: random.Random) -> dict:
    """
    Chuyển đổi 1 raw HotpotQA item sang schema QAExample dict.

    Context: giữ các paragraph có trong supporting_facts
    + thêm 1-2 distractor ngẫu nhiên.
    """
    qid: str = item["_id"]
    question: str = item["question"]
    gold_answer: str = item["answer"]
    difficulty: str = _level_to_difficulty(item.get("level", "medium"))

    # Xây dựng mapping title -> text
    all_paragraphs: dict[str, str] = {}
    for title, sentences in item.get("context", []):
        all_paragraphs[title] = " ".join(sentences)

    # Tìm title xuất hiện trong supporting_facts
    supporting_titles: set[str] = {fact[0] for fact in item.get("supporting_facts", [])}

    context_chunks: list[dict] = []
    for title in supporting_titles:
        if title in all_paragraphs:
            context_chunks.append({"title": title, "text": all_paragraphs[title]})

    # Thêm 1-2 distractor ngẫu nhiên
    distractor_titles = [t for t in all_paragraphs if t not in supporting_titles]
    num_distractors = rng.randint(1, min(2, len(distractor_titles))) if distractor_titles else 0
    chosen_distractors = rng.sample(distractor_titles, num_distractors)
    for title in chosen_distractors:
        context_chunks.append({"title": title, "text": all_paragraphs[title]})

    # Shuffle để không để lộ thứ tự
    rng.shuffle(context_chunks)

    return {
        "qid": qid,
        "difficulty": difficulty,
        "question": question,
        "gold_answer": gold_answer,
        "context": context_chunks,
    }


@app.command()
def main(
    n: int = typer.Option(120, help="Số mẫu cần chọn"),
    seed: int = typer.Option(42, help="Random seed để reproducible"),
    out: str = typer.Option("data/hotpot_real_120.json", help="Đường dẫn output"),
    cache: str = typer.Option(str(DEFAULT_CACHE), help="Cache path cho raw file"),
) -> None:
    """Tải HotpotQA dev-distractor và xuất ra file JSON schema QAExample."""
    out_path = Path(out)
    cache_path = Path(cache)

    raw_items = download_hotpotqa(cache_path)
    console.log(f"Tổng mẫu gốc: {len(raw_items)}")

    rng = random.Random(seed)
    sampled = rng.sample(raw_items, min(n, len(raw_items)))
    console.log(f"Đã chọn {len(sampled)} mẫu (seed={seed})")

    qa_examples = [convert_item(item, rng) for item in sampled]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(qa_examples, ensure_ascii=False, indent=2), encoding="utf-8")
    console.log(f"[green]Đã lưu:[/green] {out_path} ({len(qa_examples)} mẫu)")


if __name__ == "__main__":
    app()
