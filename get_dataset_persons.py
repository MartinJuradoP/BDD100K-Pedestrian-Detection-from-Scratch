import os
import json
from PIL import Image
import shutil
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# Load a JSON annotation file from BDD100K
def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# Extract all bounding boxes for the 'person' category from annotation data
def get_person_boxes(annotations):
    boxes = []
    for frame in annotations.get("frames", []):
        for obj in frame.get("objects", []):
            if obj.get("category") == "person":
                box = obj.get("box2d")
                if box and all(k in box for k in ["x1", "y1", "x2", "y2"]):
                    boxes.append((box["x1"], box["y1"], box["x2"], box["y2"]))
    return boxes

# Clean and recreate output folders for a specific subset (e.g., train/val/test)
def clean_folders(output_base_dir, subset):
    for category in ["with_people", "without_people"]:
        path = os.path.join(output_base_dir, category, subset)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    print("🧹 Folders have been cleaned and recreated.")

# Process a single file (image + annotation):
# - Load image and JSON annotation
# - Check if it contains any 'person' bounding box with a significant area
# - Save image to the appropriate folder based on result
def process_file(json_file, label_subdir, image_subdir, out_with, out_without, size_threshold):
    json_path = os.path.join(label_subdir, json_file)
    annotations = load_json(json_path)
    image_name = f"{annotations['name']}.jpg"
    image_path = os.path.join(image_subdir, image_name)

    if not os.path.exists(image_path):
        return ("not_found", image_name)

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return ("error", image_name)

    boxes = get_person_boxes(annotations)

    # If there are no person annotations, classify as "without people"
    if not boxes:
        image.save(os.path.join(out_without, image_name))
        return ("without_people", image_name)

    # Calculate total image area
    image_area = image.width * image.height

    # Check if any person occupies more than the threshold area
    valid_person = any(
        ((x2 - x1) * (y2 - y1)) / image_area >= size_threshold
        for (x1, y1, x2, y2) in boxes
    )

    # Save the image in the appropriate folder
    if valid_person:
        image.save(os.path.join(out_with, image_name))
        return ("with_people", image_name)
    else:
        # Ignore images where all persons are too small (avoid noise or bias)
        return ("ignored", image_name)

# Balance the dataset so both classes have the same number of images
def balance_dataset(out_with, out_without):
    with_people_files = os.listdir(out_with)
    without_people_files = os.listdir(out_without)

    min_count = min(len(with_people_files), len(without_people_files))

    random.shuffle(with_people_files)
    random.shuffle(without_people_files)

    for file in with_people_files[min_count:]:
        os.remove(os.path.join(out_with, file))

    for file in without_people_files[min_count:]:
        os.remove(os.path.join(out_without, file))

    print(f"\n🔁 Dataset balanced: {min_count} images per class (with_people/without_people)")

# Full pipeline for processing a given dataset subset (train/val/test)
def process_dataset(label_dir, image_dir, output_base_dir, subset='train', size_threshold=0.03):
    clean_folders(output_base_dir, subset)

    label_subdir = os.path.join(label_dir, subset)
    image_subdir = os.path.join(image_dir, subset)

    out_with = os.path.join(output_base_dir, "with_people", subset)
    out_without = os.path.join(output_base_dir, "without_people", subset)

    json_files = [f for f in os.listdir(label_subdir) if f.endswith(".json")]

    results = {
        "with_people": 0,
        "without_people": 0,
        "ignored": 0,
        "error": 0,
        "not_found": 0
    }

    # Use multiprocessing to speed up the file processing
    with ProcessPoolExecutor() as executor:
        tasks = [executor.submit(process_file, f, label_subdir, image_subdir, out_with, out_without, size_threshold)
                 for f in json_files]

        for future in as_completed(tasks):
            result, _ = future.result()
            if result in results:
                results[result] += 1

    # Print processing summary
    print(f"\n✅ Processing completed for subset: {subset}")
    print(f"📦 with_people (≥ {int(size_threshold * 100)}% area): {results['with_people']}")
    print(f"📦 without_people (no 'person' category): {results['without_people']}")
    print(f"📁 ignored (people < {int(size_threshold * 100)}% area): {results['ignored']}")
    print(f"❌ errors: {results['error']}")
    print(f"⚠️ not found: {results['not_found']}")

    # Balance both classes to prevent model bias
    balance_dataset(out_with, out_without)

# Main entry point to define subset and person visibility threshold
if __name__ == "__main__":
    process_dataset(
        label_dir="data/labels",         # Directory with JSON annotations (BDD100K)
        image_dir="data/images",         # Directory with images (BDD100K)
        output_base_dir="bdd_people",    # Output directory for the processed dataset
        subset="train",                  # Subset to process: 'train', 'val', etc.
        size_threshold=0.005             # Minimum person area (fraction of image)
    )