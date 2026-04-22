"""One-off inspector for Gemini 3.x thought traces on recorded sub-tasks.

Usage:
    export GOOGLE_API_KEY=...
    python scripts/inspect_thoughts.py \
        --sub-task packing_order_chopsticks --doc-ids 0 1 2 \
        --model gemini-3.1-pro-preview

Reads the same merged_qa JSON used by the eval harness, rebuilds the prompt
via lmms_eval.tasks.longvid-reasoning-eval_recorded.utils, calls Gemini with
includeThoughts=True, and prints thought-summary parts and the final answer
separately. Does not touch the eval harness or saved logs.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_JSON = "/nas2/benchmarks/vpi/recorded/merged_qa/all.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UTILS_PATH = REPO_ROOT / "lmms_eval/tasks/longvid-reasoning-eval_recorded/utils.py"


def load_utils(utils_path):
    spec = importlib.util.spec_from_file_location("task_utils", utils_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=DEFAULT_JSON, help="merged_qa JSON path")
    p.add_argument("--utils-path", default=str(DEFAULT_UTILS_PATH),
                   help="lmms_eval task utils.py to load process_*_docs / doc_to_text / doc_to_visual")
    p.add_argument("--sub-task", required=True, help="e.g. packing_order_chopsticks")
    p.add_argument("--doc-ids", type=int, nargs="+", default=[0], help="doc indices to inspect")
    p.add_argument("--model", default="gemini-3.1-pro-preview")
    p.add_argument("--max-tokens", type=int, default=65536)
    p.add_argument("--out-dir", default=None,
                   help="If set, write one .txt per (sub_task, doc_id) with full thought + answer")
    return p.parse_args()


def build_doc(utils_mod, raw_entry, sub_task):
    """Mimic process_*_docs for a single entry. Returns dict passed to doc_to_text/doc_to_visual."""
    # The process_<sub_task>_docs functions wrap raw entries into the doc schema.
    proc = getattr(utils_mod, f"process_{sub_task}_docs")
    ds = proc([{"data": {sub_task: [raw_entry]}}])
    return ds[0]


def upload_video(client, video_path):
    uploaded = client.files.upload(file=video_path)
    while uploaded.state == "PROCESSING":
        time.sleep(3)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state == "FAILED":
        raise RuntimeError(f"Video upload failed: {video_path}")
    return uploaded


def main():
    args = parse_args()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: set GOOGLE_API_KEY", file=sys.stderr)
        sys.exit(1)

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    utils_mod = load_utils(args.utils_path)
    data = json.load(open(args.json))
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    entries = data["data"].get(args.sub_task)
    if not entries:
        print(f"no sub-task {args.sub_task} in {args.json}", file=sys.stderr)
        sys.exit(1)

    for did in args.doc_ids:
        if did >= len(entries):
            print(f"doc_id {did} out of range (max {len(entries)-1})")
            continue
        raw = entries[did]
        doc = build_doc(utils_mod, raw, args.sub_task)
        prompt = utils_mod.doc_to_text(doc)
        video_path = utils_mod.doc_to_visual(doc)[0]
        gt = doc.get("answer_text")
        target_value = doc.get("target_value")

        print("=" * 100)
        print(f"sub_task={args.sub_task}  doc_id={did}  GT={gt}  target_value={target_value}")
        print(f"video: {video_path}")
        print("-" * 100)
        print("PROMPT:")
        print(prompt)
        print("-" * 100)

        uploaded = upload_video(client, video_path)
        config = types.GenerateContentConfig(
            maxOutputTokens=args.max_tokens,
            temperature=0,
            thinkingConfig=types.ThinkingConfig(includeThoughts=True),
            safetySettings=[
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            ],
        )
        response = client.models.generate_content(
            model=args.model,
            contents=[prompt, uploaded],
            config=config,
        )

        usage = response.usage_metadata
        print(
            f"tokens: input={usage.prompt_token_count}  "
            f"output={usage.candidates_token_count}  "
            f"reasoning={getattr(usage, 'thoughts_token_count', 0)}"
        )
        print("-" * 100)
        if response.candidates and response.candidates[0].content:
            thought_parts = []
            answer_parts = []
            for part in response.candidates[0].content.parts:
                if getattr(part, "thought", False):
                    thought_parts.append(part.text or "")
                elif part.text:
                    answer_parts.append(part.text)
            thought_text = "\n".join(thought_parts) if thought_parts else "(no thought summary returned by API)"
            final_text = "".join(answer_parts)
            print("THOUGHT SUMMARY:")
            print(thought_text)
            print("-" * 100)
            print("FINAL ANSWER:")
            print(final_text)
            if out_dir:
                out_file = out_dir / f"{args.sub_task}__doc{did}.txt"
                out_file.write_text(
                    f"sub_task={args.sub_task} doc_id={did} GT={gt} target_value={target_value}\n"
                    f"video={video_path}\n"
                    f"tokens input={usage.prompt_token_count} output={usage.candidates_token_count} reasoning={getattr(usage,'thoughts_token_count',0)}\n"
                    f"------ PROMPT ------\n{prompt}\n"
                    f"------ THOUGHT SUMMARY ------\n{thought_text}\n"
                    f"------ FINAL ANSWER ------\n{final_text}\n"
                )
        print()

        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass


if __name__ == "__main__":
    main()
