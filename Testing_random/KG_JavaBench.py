import json, argparse, pathlib
from kg_workflow import KgWorkflow   # <-- your existing workflow class/module

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to JavaBench dataset (e.g., datasets/selective-context/data-PA19.jsonl)")
    ap.add_argument("--output", required=True, help="Where to save samples.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of samples (for quick testing)")
    args = ap.parse_args()

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.data, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            sample = json.loads(line)
            task_id = sample["task_id"]
            target = sample["target"]   # the ground-truth file/class
            incomplete_code = sample["code"]  # has // TODO

            # 🔹 Call your workflow
            completion = KgWorkflow(incomplete_code)

            record = {
                "task_id": task_id,
                "prompt": incomplete_code,
                "target": target,
                "completion": completion
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if args.limit and i + 1 >= args.limit:
                break

    print(f"Done. Wrote completions to {out_path}")

if __name__ == "__main__":
    main()
