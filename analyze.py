import json
from collections import Counter, defaultdict

with open('results/knowledge_graph_final.json', encoding='utf-8') as f:
    kg = json.load(f)

ents = {e['id']: e['label'] for e in kg['entities']}

print('=== SAMPLE TRIPLETS ===')
for r in kg['relations'][:30]:
    s = ents.get(r['subject_id'], r['subject_id'])
    o = ents.get(r['object_id'], r['object_id'])
    p = r['predicate']
    print(f"{s} --[{p}]--> {o}")

print()
print('=== ENTITY TYPES ===')
types = Counter(e['entity_type'] for e in kg['entities'])
for t, n in types.most_common():
    print(f"{t}: {n}")

print()
print('=== TOP ENTITIES BY CONNECTIONS ===')
degree = defaultdict(int)
for r in kg['relations']:
    degree[r['subject_id']] += 1
    degree[r['object_id']] += 1
top = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:15]
for eid, deg in top:
    label = ents.get(eid, eid)
    print(f"{label}: {deg} connections")

print()
print('=== STATS ===')
print(f"Total entities: {len(kg['entities'])}")
print(f"Total relations: {len(kg['relations'])}")