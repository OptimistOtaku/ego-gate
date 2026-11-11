"""Summarize observed log progress; never use this as a completeness audit."""
import argparse
import json
from collections import Counter
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('log',type=Path)
a=p.parse_args()
seen={}
for line in a.log.read_text(encoding='utf-8',errors='replace').splitlines():
    try:
        row=json.loads(line)
        key=tuple(row[k] for k in ('dataset','seed','budget','method'))
        seen[key]=row
    except (ValueError,KeyError,TypeError):
        continue
print(json.dumps(dict(observed_completed_rows=len(seen),expected=870,
                      by_dataset=dict(Counter(k[0] for k in seen)),
                      full_completion_message='GPU benchmark complete' in a.log.read_text(encoding='utf-8',errors='replace'),
                      note='Live-log count only; downloaded artifacts and audit determine completion'),indent=2))
