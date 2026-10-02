"""
merge_datasets.py
=================
Merges three fracture datasets into a single unified YOLO-format dataset:
  1. BoneFractureYolo8  (7 classes → remapped)
  2. FracAtlas          (1 class → split by anatomy using dataset.csv)
  3. GRAZPEDWRI-DX      (9 classes → only 'fracture' class kept)

Unified 9-class schema:
  0: elbow_fracture
  1: finger_fracture
  2: forearm_fracture
  3: humerus_fracture
  4: shoulder_fracture
  5: wrist_fracture
  6: hand_fracture
  7: hip_fracture
  8: leg_fracture

Usage (on Kaggle):
  python merge_datasets.py

Outputs to: /kaggle/working/merged_dataset/
"""

import os, shutil, zipfile, random, yaml, csv
from pathlib import Path
from collections import Counter, defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

KAGGLE_INPUT = Path('/kaggle/input')
WORKING_DIR  = Path('/kaggle/working')
MERGED_DIR   = WORKING_DIR / 'merged_dataset'
MERGED_YAML  = WORKING_DIR / 'merged_data.yaml'

# Unified class definitions
UNIFIED_CLASSES = [
    'elbow_fracture',    # 0
    'finger_fracture',   # 1
    'forearm_fracture',  # 2
    'humerus_fracture',  # 3
    'shoulder_fracture', # 4
    'wrist_fracture',    # 5
    'hand_fracture',     # 6
    'hip_fracture',      # 7
    'leg_fracture',      # 8
]
NC = len(UNIFIED_CLASSES)

# BoneFractureYolo8 class remap
# Old: 0=elbow pos, 1=fingers pos, 2=forearm frac, 3=humerus frac,
#      4=humerus, 5=shoulder frac, 6=wrist pos
BONE8_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 5}

# GRAZPEDWRI-DX: only keep class 3 (fracture), remap → 5 (wrist_fracture)
# It's a pediatric wrist dataset → all fractures are wrist fractures
GRAZ_FRACTURE_CLASS = 3
GRAZ_TARGET_CLASS   = 5  # wrist_fracture

# FracAtlas: body_part string → unified class id
FRACATLAS_BODYPART_MAP = {
    'Hand'    : 6,  # hand_fracture
    'Hip'     : 7,  # hip_fracture
    'Leg'     : 8,  # leg_fracture
    'Shoulder': 4,  # shoulder_fracture
}

# Train/val/test split ratio for FracAtlas (it has no pre-made split)
FRACATLAS_SPLIT = (0.75, 0.15, 0.10)
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def copy_image(src: Path, dst_dir: Path):
    """Copy an image to dst_dir, return dest path."""
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst

def write_label(dst_dir: Path, stem: str, lines: list[str]):
    """Write YOLO label file. If lines is empty writes an empty file (background)."""
    label_path = dst_dir / f'{stem}.txt'
    label_path.write_text('\n'.join(lines))

def remap_label_file(src_lbl: Path, class_remap: dict) -> list[str]:
    """Read a YOLO label file and remap class IDs. Returns list of new lines."""
    out = []
    if not src_lbl.exists():
        return out
    for line in src_lbl.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        if cls in class_remap:
            parts[0] = str(class_remap[cls])
            out.append(' '.join(parts))
        # lines with unmapped classes are silently dropped
    return out

def filter_label_file(src_lbl: Path, keep_class: int, new_class: int) -> list[str]:
    """Keep only lines with keep_class, remap to new_class."""
    out = []
    if not src_lbl.exists():
        return out
    for line in src_lbl.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if int(parts[0]) == keep_class:
            parts[0] = str(new_class)
            out.append(' '.join(parts))
    return out

def find_image(img_dir: Path, stem: str) -> Path | None:
    """Find image file by stem (try jpg, jpeg, png)."""
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        p = img_dir / f'{stem}{ext}'
        if p.exists():
            return p
    return None

counter_total = Counter()  # global stats

def add_sample(img_src: Path, label_lines: list[str], split: str,
               dataset_tag: str):
    """Copy image + write label into merged dataset split."""
    img_dst_dir = MERGED_DIR / split / 'images'
    lbl_dst_dir = MERGED_DIR / split / 'labels'

    # Avoid filename collisions by prefixing with dataset tag
    new_stem = f'{dataset_tag}_{img_src.stem}'
    dst_img = img_dst_dir / f'{new_stem}{img_src.suffix}'
    shutil.copy2(img_src, dst_img)
    write_label(lbl_dst_dir, new_stem, label_lines)

    for line in label_lines:
        cls = int(line.split()[0])
        counter_total[UNIFIED_CLASSES[cls]] += 1

# ──────────────────────────────────────────────────────────────────────────────
# DATASET 1: BoneFractureYolo8
# ──────────────────────────────────────────────────────────────────────────────

def process_bone8(extract_dir: Path):
    print('\n' + '='*60)
    print('Processing BoneFractureYolo8 ...')
    count = 0
    for split in ['train', 'valid', 'test']:
        # valid → val mapping
        out_split = 'val' if split == 'valid' else split
        img_dir = extract_dir / split / 'images'
        lbl_dir = extract_dir / split / 'labels'
        if not img_dir.exists():
            print(f'  WARNING: {img_dir} not found, skipping.')
            continue
        imgs = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.png'))
        for img in imgs:
            lbl = lbl_dir / f'{img.stem}.txt'
            new_lines = remap_label_file(lbl, BONE8_REMAP)
            # Even if empty (background), include the image
            add_sample(img, new_lines, out_split, 'b8')
            count += 1
        print(f'  {split} → {out_split}: {len(imgs)} images')
    print(f'  BoneFractureYolo8 total: {count} images')

# ──────────────────────────────────────────────────────────────────────────────
# DATASET 2: FracAtlas
# ──────────────────────────────────────────────────────────────────────────────

def extract_fracatlas(zip_path: Path) -> Path:
    """Extract FracAtlas zip and return root dir."""
    extract_to = WORKING_DIR / 'FracAtlas_extracted'
    if extract_to.exists():
        print('  FracAtlas already extracted.')
        return extract_to
    print(f'  Extracting {zip_path} ...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print('  Extraction complete.')
    return extract_to

def find_fracatlas_root(extracted: Path) -> Path:
    """Find the actual FracAtlas root (handles nested zip structure)."""
    # Could be extracted/FracAtlas/ or extracted/ directly
    for candidate in [extracted / 'FracAtlas', extracted]:
        if (candidate / 'dataset.csv').exists():
            return candidate
        # Search one level deeper
        for sub in candidate.iterdir():
            if sub.is_dir() and (sub / 'dataset.csv').exists():
                return sub
    raise FileNotFoundError(f'Cannot find dataset.csv inside {extracted}')

def process_fracatlas(zip_path: Path):
    print('\n' + '='*60)
    print('Processing FracAtlas ...')

    extracted = extract_fracatlas(zip_path)
    root = find_fracatlas_root(extracted)
    print(f'  FracAtlas root: {root}')

    # Read dataset.csv  ─ columns: image_id, fractured, body_part, ...
    csv_path = root / 'dataset.csv'
    image_meta = {}  # image_id → {fractured, body_part}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalise key names (they vary slightly across versions)
            img_id     = row.get('image_id') or row.get('Image ID') or row.get('id')
            fractured  = str(row.get('fractured', row.get('Fractured', '0'))).strip()
            body_part  = str(row.get('body_part', row.get('Body Part',
                             row.get('anatomy', '')))).strip()
            if img_id:
                image_meta[img_id.strip()] = {
                    'fractured' : fractured in ('1', 'True', 'true', 'yes'),
                    'body_part' : body_part,
                }

    print(f'  CSV rows loaded: {len(image_meta)}')

    # Find YOLO annotation directory
    yolo_lbl_dir = None
    for candidate in [
        root / 'Annotations' / 'YOLO',
        root / 'annotations' / 'yolo',
        root / 'labels',
        root / 'YOLO',
    ]:
        if candidate.exists():
            yolo_lbl_dir = candidate
            break
    if yolo_lbl_dir is None:
        print('  WARNING: No YOLO annotation dir found in FracAtlas. Skipping.')
        return

    print(f'  YOLO label dir: {yolo_lbl_dir}')

    # Find all fractured images (images that have bboxes)
    # Images live in images/Fractured/ and images/Non-Fractured/
    img_root = root / 'images'
    all_imgs = {}  # stem → Path
    for img in img_root.rglob('*.jpg'):
        all_imgs[img.stem] = img
    for img in img_root.rglob('*.jpeg'):
        all_imgs[img.stem] = img
    for img in img_root.rglob('*.png'):
        all_imgs[img.stem] = img

    print(f'  Total images found: {len(all_imgs)}')

    # Build list of (img_path, unified_class, lbl_path) for fractured images only
    samples = []  # (img_path, new_class_id, lbl_path_or_None)

    for stem, img_path in all_imgs.items():
        meta = image_meta.get(stem)
        if meta is None:
            # Try matching by filename without extension
            meta = image_meta.get(stem.split('_')[0])  # some versions store base ids

        # Determine body part: from CSV or from folder name
        if meta and meta['fractured']:
            body_part = meta['body_part']
        else:
            # Infer from parent folder name (Fractured sub-folders named by anatomy)
            parent = img_path.parent.name  # e.g. 'Fractured', 'Hand', 'Hip'
            body_part = parent if parent in FRACATLAS_BODYPART_MAP else ''
            if not body_part:
                continue  # skip non-fractured or unidentifiable

        unified_cls = FRACATLAS_BODYPART_MAP.get(body_part)
        if unified_cls is None:
            # body_part might be lowercase
            unified_cls = FRACATLAS_BODYPART_MAP.get(body_part.capitalize())
        if unified_cls is None:
            continue  # unknown anatomy, skip

        lbl_path = yolo_lbl_dir / f'{stem}.txt'
        samples.append((img_path, unified_cls, lbl_path))

    print(f'  Fractured samples with known anatomy: {len(samples)}')

    # Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * FRACATLAS_SPLIT[0])
    n_val   = int(n * FRACATLAS_SPLIT[1])

    split_map = (
        [('train', samples[:n_train]),
         ('val',   samples[n_train:n_train+n_val]),
         ('test',  samples[n_train+n_val:])]
    )

    for split_name, subset in split_map:
        for img_path, unified_cls, lbl_path in subset:
            # Remap the label: original class 0 (fractured) → unified_cls
            if lbl_path.exists():
                new_lines = []
                for line in lbl_path.read_text().strip().splitlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        parts[0] = str(unified_cls)
                        new_lines.append(' '.join(parts))
            else:
                # No bbox annotation found — create a note but still add image
                # as background (empty label)
                new_lines = []
            add_sample(img_path, new_lines, split_name, 'fa')
        print(f'  {split_name}: {len(subset)} images')

    print(f'  FracAtlas total: {n} fractured images processed')

# ──────────────────────────────────────────────────────────────────────────────
# DATASET 3: GRAZPEDWRI-DX
# ──────────────────────────────────────────────────────────────────────────────

def find_graz_root() -> Path | None:
    """Auto-detect GRAZPEDWRI-DX dataset under /kaggle/input/."""
    for candidate in KAGGLE_INPUT.rglob('*'):
        if not candidate.is_dir():
            continue
        if 'graz' in candidate.name.lower():
            return candidate
    for subdir in KAGGLE_INPUT.iterdir():
        if (subdir / 'images').exists() and (subdir / 'labels').exists():
            if 'graz' in subdir.name.lower():
                return subdir
    return None


def find_graz_yolo_labels(root: Path) -> Path | None:
    """
    Find the YOLO label dir in the raw Figshare GRAZPEDWRI-DX layout.
    Labels live inside folder_structure/ (e.g. folder_structure/labels/ or
    folder_structure/yolov5/).
    """
    fs = root / 'folder_structure'
    if fs.exists():
        for candidate in [fs / 'labels', fs / 'yolov5', fs / 'annotations']:
            if candidate.is_dir() and any(candidate.glob('*.txt')):
                return candidate
        # Recursive search inside folder_structure
        for candidate in fs.rglob('labels'):
            if candidate.is_dir() and any(candidate.glob('*.txt')):
                return candidate
    # Generic fallback search
    for candidate in root.rglob('labels'):
        if candidate.is_dir() and any(candidate.glob('*.txt')):
            return candidate
    return None


def collect_graz_images(root: Path) -> dict:
    """
    Collect all images from images_part1..N folders (raw Figshare layout).
    Returns dict: stem -> Path
    """
    imgs = {}
    for part_dir in sorted(root.glob('images_part*')):
        if part_dir.is_dir():
            for img in part_dir.rglob('*'):
                if img.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    imgs[img.stem] = img
    # Also handle a plain images/ folder
    plain = root / 'images'
    if plain.exists():
        for img in plain.rglob('*'):
            if img.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                imgs[img.stem] = img
    return imgs


def find_graz_split_dirs(root: Path) -> dict:
    """Pre-split YOLO layout: images/{split}/ + labels/{split}/ or {split}/images/."""
    splits = {}
    img_base = root / 'images'
    lbl_base = root / 'labels'
    if img_base.exists() and lbl_base.exists():
        for sp in ['train', 'valid', 'val', 'test']:
            img_d = img_base / sp
            if img_d.exists():
                splits['val' if sp == 'valid' else sp] = (img_d, lbl_base / sp)
        if splits:
            return splits
    for sp in ['train', 'valid', 'val', 'test']:
        img_d = root / sp / 'images'
        if img_d.exists():
            splits['val' if sp == 'valid' else sp] = (img_d, root / sp / 'labels')
    if splits:
        return splits
    flat_imgs = (list(root.rglob('*.jpg')) + list(root.rglob('*.jpeg'))
                 + list(root.rglob('*.png')))
    lbl_d = next((d for d in [root / 'labels', root / 'annotations'] if d.is_dir()), root)
    if flat_imgs:
        splits['_flat'] = (None, lbl_d, flat_imgs)
    return splits


def process_grazpedwri(graz_root: Path):
    print('\n' + '='*60)
    print(f'Processing GRAZPEDWRI-DX from: {graz_root}')

    has_parts = any(graz_root.glob('images_part*'))
    if has_parts:
        print('  Layout: Raw Figshare (images_part* + folder_structure + dataset.csv)')
        _process_graz_raw(graz_root)
    else:
        print('  Layout: Pre-split YOLO-ready')
        _process_graz_presplit(graz_root)


def _process_graz_raw(root: Path):
    """Handle raw Figshare layout: images_part1-4, YOLO labels in folder_structure/, dataset.csv."""

    all_imgs = collect_graz_images(root)
    print(f'  Total images across all parts: {len(all_imgs):,}')

    lbl_dir = find_graz_yolo_labels(root)
    if lbl_dir is None:
        print('\n  ⚠️  No YOLO .txt label files found in folder_structure/.')
        print('  This Kaggle version only has images + dataset.csv (no bbox annotations).')
        print('  → Switch to nathann/graz-yolo-ready-v11 which has pre-made YOLO labels.')
        print('  Skipping GRAZPEDWRI-DX.')
        return
    print(f'  YOLO label dir: {lbl_dir}')

    # Patient-level split using dataset.csv
    csv_path = root / 'dataset.csv'
    patient_map = {}
    if csv_path.exists():
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stem = (row.get('filestem') or row.get('filename') or
                        row.get('image_id') or '').strip()
                pid  = (row.get('patient_id') or row.get('patientid') or stem).strip()
                if stem:
                    patient_map[stem] = pid
        print(f'  Patient CSV loaded: {len(patient_map):,} entries')

    unique_pids = list(set(patient_map.values())) if patient_map else list(all_imgs.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(unique_pids)
    n  = len(unique_pids)
    train_pids = set(unique_pids[:int(n * 0.75)])
    val_pids   = set(unique_pids[int(n * 0.75):int(n * 0.90)])

    def get_split(stem):
        pid = patient_map.get(stem, stem)
        if pid in train_pids: return 'train'
        if pid in val_pids:   return 'val'
        return 'test'

    counts = {'train': 0, 'val': 0, 'test': 0}
    kept   = {'train': 0, 'val': 0, 'test': 0}
    for stem, img_path in all_imgs.items():
        lbl_path = lbl_dir / f'{stem}.txt'
        lines    = filter_label_file(lbl_path, GRAZ_FRACTURE_CLASS, GRAZ_TARGET_CLASS)
        sp       = get_split(stem)
        add_sample(img_path, lines, sp, 'gz')
        counts[sp] += 1
        if lines:
            kept[sp] += 1

    for sp in ['train', 'val', 'test']:
        print(f'  {sp:5s}: {counts[sp]:6,} images | {kept[sp]:6,} with fractures')
    print(f'  ✅ GRAZPEDWRI-DX total: {sum(counts.values()):,} images')


def _process_graz_presplit(root: Path):
    """Handle pre-split YOLO-ready layout."""
    split_dirs = find_graz_split_dirs(root)
    if not split_dirs:
        print('  WARNING: No images found. Skipping.')
        return
    total = 0
    if '_flat' in split_dirs:
        _, lbl_dir, flat_imgs = split_dirs['_flat']
        random.seed(RANDOM_SEED)
        random.shuffle(flat_imgs)
        n  = len(flat_imgs)
        nt = int(n * 0.75)
        nv = int(n * 0.15)
        manual_splits = [('train', flat_imgs[:nt]),
                         ('val',   flat_imgs[nt:nt+nv]),
                         ('test',  flat_imgs[nt+nv:])]
        for sp, imgs in manual_splits:
            kept = 0
            for img in imgs:
                lines = filter_label_file(lbl_dir / f'{img.stem}.txt',
                                          GRAZ_FRACTURE_CLASS, GRAZ_TARGET_CLASS)
                add_sample(img, lines, sp, 'gz')
                total += 1
                if lines: kept += 1
            print(f'  {sp}: {len(imgs):,} images | {kept:,} with fractures')
    else:
        for sp, (img_dir, lbl_dir) in split_dirs.items():
            imgs = (list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg'))
                    + list(img_dir.glob('*.png')))
            kept = 0
            for img in imgs:
                lines = filter_label_file(lbl_dir / f'{img.stem}.txt',
                                          GRAZ_FRACTURE_CLASS, GRAZ_TARGET_CLASS)
                add_sample(img, lines, sp, 'gz')
                total += 1
                if lines: kept += 1
            print(f'  {sp}: {len(imgs):,} images | {kept:,} with fractures')
    print(f'  ✅ GRAZPEDWRI-DX total: {total:,} images')
    """
    Returns dict: {'train': (img_dir, lbl_dir), 'val': ..., 'test': ...}
    Handles two common layouts:
      Layout A: root/images/train/, root/labels/train/
      Layout B: root/train/images/, root/train/labels/
    """
    splits = {}

    # Layout A: images/{split} / labels/{split}
    img_base = root / 'images'
    lbl_base = root / 'labels'
    if img_base.exists() and lbl_base.exists():
        for sp in ['train', 'valid', 'val', 'test']:
            img_d = img_base / sp
            lbl_d = lbl_base / sp
            if img_d.exists():
                out_sp = 'val' if sp == 'valid' else sp
                splits[out_sp] = (img_d, lbl_d)
        if splits:
            return splits

    # Layout B: {split}/images / {split}/labels
    for sp in ['train', 'valid', 'val', 'test']:
        img_d = root / sp / 'images'
        lbl_d = root / sp / 'labels'
        if img_d.exists():
            out_sp = 'val' if sp == 'valid' else sp
            splits[out_sp] = (img_d, lbl_d)
    if splits:
        return splits

    # Layout C: flat — all images and labels in root (no split)
    # Will do a manual split
    flat_imgs = list(root.rglob('*.jpg')) + list(root.rglob('*.jpeg')) + list(root.rglob('*.png'))
    flat_lbls_dir = next((d for d in [root / 'labels', root / 'annotations', root] if d.is_dir()), root)
    if flat_imgs:
        splits['_flat'] = (None, flat_lbls_dir, flat_imgs)
    return splits

def process_grazpedwri(graz_root: Path):
    print('\n' + '='*60)
    print(f'Processing GRAZPEDWRI-DX from: {graz_root}')

    split_dirs = find_graz_split_dirs(graz_root)

    if not split_dirs:
        print('  WARNING: Could not determine GRAZPEDWRI-DX structure. Skipping.')
        return

    total = 0

    if '_flat' in split_dirs:
        # Flat layout: we split manually
        _, lbl_dir, all_imgs = split_dirs['_flat']
        random.seed(RANDOM_SEED)
        random.shuffle(all_imgs)
        n = len(all_imgs)
        n_train = int(n * 0.75)
        n_val   = int(n * 0.15)
        manual_splits = [
            ('train', all_imgs[:n_train]),
            ('val',   all_imgs[n_train:n_train+n_val]),
            ('test',  all_imgs[n_train+n_val:]),
        ]
        for split_name, imgs in manual_splits:
            kept = 0
            for img in imgs:
                lbl = lbl_dir / f'{img.stem}.txt'
                lines = filter_label_file(lbl, GRAZ_FRACTURE_CLASS, GRAZ_TARGET_CLASS)
                add_sample(img, lines, split_name, 'gz')
                total += 1
                if lines:
                    kept += 1
            print(f'  {split_name}: {len(imgs)} images, {kept} with fractures')
    else:
        for split_name, (img_dir, lbl_dir) in split_dirs.items():
            imgs = (list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg'))
                    + list(img_dir.glob('*.png')))
            kept = 0
            for img in imgs:
                lbl = lbl_dir / f'{img.stem}.txt'
                lines = filter_label_file(lbl, GRAZ_FRACTURE_CLASS, GRAZ_TARGET_CLASS)
                add_sample(img, lines, split_name, 'gz')
                total += 1
                if lines:
                    kept += 1
            print(f'  {split_name}: {len(imgs)} images, {kept} with fractures')

    print(f'  GRAZPEDWRI-DX total: {total} images')

# ──────────────────────────────────────────────────────────────────────────────
# WRITE YAML
# ──────────────────────────────────────────────────────────────────────────────

def write_yaml():
    config = {
        'path'  : str(MERGED_DIR),
        'train' : 'train/images',
        'val'   : 'val/images',
        'test'  : 'test/images',
        'nc'    : NC,
        'names' : UNIFIED_CLASSES,
    }
    with open(MERGED_YAML, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f'\ndata.yaml written to: {MERGED_YAML}')
    print(MERGED_YAML.read_text())

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print('='*60)
    print('FRACTURE DATASET MERGER')
    print('='*60)

    # Create output dirs
    for split in ['train', 'val', 'test']:
        mkdir(MERGED_DIR / split / 'images')
        mkdir(MERGED_DIR / split / 'labels')

    # ── Dataset 1: BoneFractureYolo8 ──────────────────────────────────────────
    bone8_dir = None
    for candidate in KAGGLE_INPUT.rglob('BoneFractureYolo8'):
        if candidate.is_dir() and (candidate / 'train').exists():
            bone8_dir = candidate
            break
    if bone8_dir:
        process_bone8(bone8_dir)
    else:
        print('\nWARNING: BoneFractureYolo8 dataset not found under /kaggle/input/')

    # ── Dataset 2: FracAtlas ───────────────────────────────────────────────────
    fracatlas_zip = None
    # Check common locations
    for candidate in [
        WORKING_DIR / 'data' / 'FracAtlas.zip',
        Path('/kaggle/input/fracatlas/FracAtlas.zip'),
        *list(KAGGLE_INPUT.rglob('FracAtlas.zip')),
        *list(KAGGLE_INPUT.rglob('fracatlas*.zip')),
    ]:
        if Path(candidate).exists():
            fracatlas_zip = Path(candidate)
            break

    if fracatlas_zip is None:
        # Maybe already extracted
        for candidate in KAGGLE_INPUT.rglob('dataset.csv'):
            if 'frac' in str(candidate).lower():
                fracatlas_zip = None
                process_fracatlas_extracted(candidate.parent)
                break
        if fracatlas_zip is None:
            print('\nWARNING: FracAtlas.zip not found. Please upload it as a Kaggle dataset input.')
    else:
        process_fracatlas(fracatlas_zip)

    # ── Dataset 3: GRAZPEDWRI-DX ──────────────────────────────────────────────
    graz_root = find_graz_root()
    if graz_root:
        process_grazpedwri(graz_root)
    else:
        print('\nWARNING: GRAZPEDWRI-DX dataset not found under /kaggle/input/')
        print('  Add dataset: jasonroggy/grazpedwri-dx')

    # ── Write YAML ────────────────────────────────────────────────────────────
    write_yaml()

    # ── Final Stats ───────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('MERGED DATASET STATS')
    print('='*60)
    for split in ['train', 'val', 'test']:
        n_img = len(list((MERGED_DIR / split / 'images').glob('*')))
        n_lbl = len(list((MERGED_DIR / split / 'labels').glob('*')))
        print(f'  {split:5s}: {n_img:6,} images | {n_lbl:6,} labels')

    print('\nAnnotation count per class (train+val+test):')
    for cls_name, cnt in sorted(counter_total.items(), key=lambda x: -x[1]):
        bar = '█' * (cnt // 100)
        print(f'  {cls_name:20s}: {cnt:6,}  {bar}')

    total_ann = sum(counter_total.values())
    print(f'\n  Total annotations: {total_ann:,}')
    print('\n✅ Merge complete! Use merged_data.yaml for training.')


if __name__ == '__main__':
    main()
