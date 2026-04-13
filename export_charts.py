import json
from collections import Counter, defaultdict

with open('results/knowledge_graph_final.json', encoding='utf-8') as f:
    kg = json.load(f)

ents = {e['id']: e['label'] for e in kg['entities']}

# Красивые примеры триплетов для диплома (фильтруем осмысленные)
print("=== BEST TRIPLETS FOR THESIS TABLE ===")
shown = set()
count = 0
for r in kg['relations']:
    s = ents.get(r['subject_id'], '')
    o = ents.get(r['object_id'], '')
    p = r['predicate']
    key = (s, p, o)
    # Берём только осмысленные (не технические)
    skip = ['C91', 'C83', 'C95', 'клинические рекомендации',
            'лечение', 'диагностика', '1,3', '0,', '/м2']
    if any(x in s or x in o for x in skip):
        continue
    if key not in shown and len(s) > 3 and len(o) > 3:
        shown.add(key)
        print(f"{s} | {p} | {o}")
        count += 1
    if count >= 15:
        break