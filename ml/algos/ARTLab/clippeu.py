import clip
import torch
from PIL import Image
import os
import openpyxl
from openpyxl import Workbook

# model clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

classes = ["trafficlight", "stop", "speedlimit", "crosswalk"]

text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)

wb = Workbook()
ws = wb.active
ws.append(["Image", "Predicted Class"])

image_folder = "RoadSign/images"
image_files = [f for f in os.listdir(image_folder)]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(1)
    predicted_class = classes[indices[0]]

    ws.append([image_file, predicted_class])

excel_file = "RoadSign/predicted_classes.xlsx"
wb.save(excel_file)

print(f"Les résultats ont été enregistrés dans {excel_file}")
