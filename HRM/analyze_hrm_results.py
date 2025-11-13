import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

path = "checkpoints/sudoku-extreme/step_0_all_preds.0"
data = torch.load(path, map_location="cpu")

inputs = data["inputs"]
labels = data["labels"]
logits = data["logits"]

# Logits → tahmine dönüştür
preds = logits.argmax(dim=-1)

correct = (preds == labels)
overall_acc = correct.float().mean().item() * 100

print(f"✅ Genel hücre doğruluğu: {overall_acc:.2f}%")

# Satır/sütun bazlı doğruluk
row_acc = correct.view(-1, 9, 9).float().mean(dim=0).mean(dim=1)  # 9 satır
col_acc = correct.view(-1, 9, 9).float().mean(dim=0).mean(dim=0)  # 9 sütun

# Rakam bazlı hata
digits = torch.arange(1,10)
digit_acc = []
for d in digits:
    mask = labels == d
    if mask.sum() > 0:
        acc = correct[mask].float().mean().item() * 100
        digit_acc.append((int(d), acc))

# Sonuçları kaydet
results = {
    "overall_accuracy": overall_acc,
    "row_accuracy": [float(x*100) for x in row_acc],
    "col_accuracy": [float(x*100) for x in col_acc],
    "digit_accuracy": {str(d): a for d, a in digit_acc}
}
os.makedirs("checkpoints/sudoku-extreme", exist_ok=True)
with open("checkpoints/sudoku-extreme/hrm_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

print("📊 Analiz kaydedildi: checkpoints/sudoku-extreme/hrm_analysis.json")

# Görselleştirme (opsiyonel)
plt.figure(figsize=(6,5))
sns.heatmap(correct.view(-1,9,9).float().mean(0).numpy(), annot=True, fmt=".2f", cmap="viridis")
plt.title("Sudoku Hücre Doğruluk Haritası (ortalama)")
plt.show()
