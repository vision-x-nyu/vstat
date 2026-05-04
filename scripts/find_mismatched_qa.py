import json
import os
import re
import glob
from collections import defaultdict

dir_path = "/nas2/willis/longvid-reasoning-eval/results/longvid-reasoning-eval_ytb/gemini3_flash/gemini-3-flash-preview/20260428_215436"

records = defaultdict(list)
for path in glob.glob(os.path.join(dir_path, "**", "*.jsonl"), recursive=True):
    task = path.split("/")[-1].split(".")[0].split("_")[-1]
    print(task)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # data['task'] = 
            records[task].append(json.loads(line))

records = dict(records)

total_qa = 0
total_mismatch_qa = 0
no_choice_qa = 0
choice_pattern = re.compile(r'\b[A-D]\b')
for task, record in records.items():
    for r in record:
        total_qa += 1
        if not r.get('mra'):
            resp = r['filtered_resps']
            resp_text = resp if isinstance(resp, str) else ' '.join(map(str, resp))
            if len(resp_text) > 50:
                print(r['doc_id'], r['video_path'])
                total_mismatch_qa += 1
            if not choice_pattern.search(resp_text):
                print('NO CHOICE:', task, r['doc_id'], r['video_path'], '->', resp_text)
                no_choice_qa += 1
print('total_qa:', total_qa)
print('total_mismatch_qa:', total_mismatch_qa)
print('no_choice_qa:', no_choice_qa)