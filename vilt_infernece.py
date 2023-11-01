from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
import time
from PIL import Image

# prepare image + question
path = "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable/000000039769.jpg"
path = "/home/projects/bagon/dannyh/data/coco_filt/val2017/Constant/000000039769.jpg"
image = Image.open(path)
text = "What are the cats laying on?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
start_time = time.time()
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
end_time = time.time()

print("Predicted answer:", model.config.id2label[idx])
print("Execution time: ", end_time - start_time)