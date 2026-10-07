import json
import glob

for nb_file in glob.glob('*.ipynb'):
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    imgsz = 800
    if nb_file == 'train.ipynb':
        imgsz = 640
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            # for train_v2 and train
            if "best_model.val(" in source and "imgsz=" in source:
                if "IMGSZ" not in source.splitlines()[0]:
                    cell['source'].insert(0, f"IMGSZ = {imgsz}\n")
            elif "best_model.predict(" in source and "imgsz=" in source:
                if "IMGSZ" not in source.splitlines()[0]:
                    cell['source'].insert(0, f"IMGSZ = {imgsz}\n")

    with open(nb_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
