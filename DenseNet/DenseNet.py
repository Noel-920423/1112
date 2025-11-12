import torch
from PIL import Image
from torchvision import models, transforms

weights = models.DenseNet121_Weights.IMAGENET1K_V1
model = models.densenet121(weights=weights).eval()

preprocess = weights.transforms()  # 正確的Resize/CenterCrop/Normalize
img = Image.open("your_image.jpg").convert("RGB")
x = preprocess(img).unsqueeze(0)  # [1,3,224,224]

with torch.no_grad():
    logits = model(x)
probs = logits.softmax(dim=1)[0]
top5 = probs.topk(5)

# 取標籤名稱 / get label names
cats = [weights.meta["categories"][i] for i in top5.indices.tolist()]
for p, c in zip(top5.values.tolist(), cats):
    print(f"{p:.3f} - {c}")
