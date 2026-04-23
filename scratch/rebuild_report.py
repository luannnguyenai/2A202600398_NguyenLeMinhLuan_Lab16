from pathlib import Path
import json
from src.reflexion_lab.schemas import RunRecord
from src.reflexion_lab.reporting import build_report, save_report

def main():
    out_dir = Path("outputs/real_run")
    react_path = out_dir / "react_runs.jsonl"
    reflex_path = out_dir / "reflexion_runs.jsonl"
    
    records = []
    for p in [react_path, reflex_path]:
        if p.exists():
            with open(p, "r") as f:
                for line in f:
                    if line.strip():
                        records.append(RunRecord.model_validate(json.loads(line)))
    
    report = build_report(records, dataset_name="hotpot_real_120.json", mode="real")
    save_report(report, out_dir)
    print(f"Báo cáo đã được cập nhật tại {out_dir}")

if __name__ == "__main__":
    main()
