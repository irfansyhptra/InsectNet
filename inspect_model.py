import torch
ckpt = torch.load('backend/artifacts/model.pth', map_location='cpu')
print("Keys in checkpoint:", ckpt.keys())
if 'model' in ckpt:
    state_dict = ckpt['model']
    print("Keys in state_dict:", list(state_dict.keys())[:10])
