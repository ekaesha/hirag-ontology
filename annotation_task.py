import json, csv, random

TYPES = {
    "1": "Drug",
    "2": "Condition",
    "3": "Procedure",
    "4": "Symptom",
    "5": "AnatomicalStructure",
    "6": "DosageRegimen",
    "7": "LabTest",
    "8": "Organization",
    "9": "Other",
}

with open("results/knowledge_graph_final.json", encoding="utf-8") as f:
    kg = json.load(f)

random.seed(42)
sample = random.sample(kg["entities"], 200)
results = []

print("\nEntity Type Annotation Tool")
print("="*50)
for t, name in TYPES.items():
    print(f"  {t} = {name}")
print("  Enter = skip | q = quit and save")
print("="*50)

for i, entity in enumerate(sample):
    label = entity["label"]
    current = entity.get("entity_type", "Unknown")
    print(f"\n[{i+1}/200] {label}")
    print(f"  System predicted: {current}")
    key = input("  Correct type (1-9 / Enter=agree / q=quit): ").strip()

    if key == "q":
        break
    elif key == "":
        # Соглашаемся с системой
        results.append({
            "label": label,
            "predicted": current,
            "ground_truth": current,
            "correct": True,
        })
    elif key in TYPES:
        gt = TYPES[key]
        results.append({
            "label": label,
            "predicted": current,
            "ground_truth": gt,
            "correct": current == gt,
        })

with open("results/annotation_ground_truth.csv", "w",
          newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["label","predicted","ground_truth","correct"]
    )
    writer.writeheader()
    writer.writerows(results)

if results:
    map1 = sum(r["correct"] for r in results) / len(results)
    print(f"\n{'='*50}")
    print(f"Annotated: {len(results)} entities")
    print(f"MAP@1 = {map1:.3f}")
    print(f"Saved: results/annotation_ground_truth.csv")
    print(f"{'='*50}")