import json

# Read classes from clases.txt
with open('backend/artifacts/clases.txt', 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f if line.strip()]

# Write labels.json
labels_dict = {str(i): cls_name for i, cls_name in enumerate(classes)}
with open('backend/artifacts/labels.json', 'w', encoding='utf-8') as f:
    json.dump(labels_dict, f, indent=2)

# Read metadata.json, update num_classes, write back
with open('backend/artifacts/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

metadata['num_classes'] = len(classes)

with open('backend/artifacts/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print(f"Updated metadata.json with num_classes = {len(classes)}")
print(f"Updated labels.json with {len(labels_dict)} classes")
