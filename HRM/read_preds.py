import torch

# Dosya adı
path = "checkpoints/sudoku-extreme/step_0_all_preds.0"

# Dosyayı yükle
data = torch.load(path, map_location="cpu")

print("📂 Anahtarlar:", list(data.keys()))

# Her anahtarın boyutlarını yazdır
for k, v in data.items():
    if hasattr(v, 'shape'):
        print(f"{k}: tensor {tuple(v.shape)}")
    else:
        print(f"{k}: {type(v)}")
