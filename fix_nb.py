import json

with open('train_v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "metrics = best.val(" in source:
            cell['source'].insert(0, "RUN = \"fracture_v3\"; IMGSZ = 800\n")
        elif "RUN_DIR = WORKING_DIR" in source:
            cell['source'].insert(0, "RUN = \"fracture_v3\"\n")
        elif "preds     = best.predict(" in source:
            cell['source'].insert(0, "IMGSZ = 800\n")

with open('train_v3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
