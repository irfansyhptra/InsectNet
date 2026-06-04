import re

with open('/home/irfan/Project/ML/Sistem/backend/tests/test_ml_service.py', 'r') as f:
    text = f.read()

text = re.sub(
    r"def exists_side_effect\(\*args, \*\*kwargs\):\n\s+if 'model.pth' in str\(self\):\n\s+return False\n\s+return True",
    "exists_side_effect = [False, True, True]",
    text
)

text = re.sub(
    r"def exists_side_effect\(\*args, \*\*kwargs\):\n\s+if 'labels.json' in str\(self\):\n\s+return False\n\s+return True",
    "exists_side_effect = [True, False, True]",
    text    
)

text = re.sub(
    r"def exists_side_effect\(\*args, \*\*kwargs\):\n\s+if 'metadata.json' in str\(self\):\n\s+return False\n\s+return True",
    "exists_side_effect = [True, True, False]",
    text
)

with open('/home/irfan/Project/ML/Sistem/backend/tests/test_ml_service.py', 'w') as f:
    f.write(text)
