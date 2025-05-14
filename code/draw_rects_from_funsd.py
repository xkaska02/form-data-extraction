from datasets import load_dataset
from PIL import Image, ImageDraw
from utils import draw_rect
import json

dataset = load_dataset("nielsr/funsd-iob-original")

with open("out/funsd_out.json", "r") as f:
    results = json.load(f)



for i in range(len(results)):
    image = dataset["test"][i]["image"]
    draw = ImageDraw.Draw(image, "RGBA")
    for j in range(len(results[i]['bboxes'])):
        print(j)
        if results[i]['bboxes'][j]:
            if results[i]['ner_tags'][j] == 'B-HEADER':
                draw.rectangle(results[i]['bboxes'][j], fill=(0,0,127,127))
            elif results[i]['ner_tags'][j] == 'I-HEADER':
                draw.rectangle(results[i]['bboxes'][j], fill=(0,0,255,127))
            elif results[i]['ner_tags'][j] == 'B-QUESTION':
                draw.rectangle(results[i]['bboxes'][j], fill=(0,127,0,127))
            elif results[i]['ner_tags'][j] == 'I-QUESTION':
                draw.rectangle(results[i]['bboxes'][j], fill=(0,255,0,127))
            elif results[i]['ner_tags'][j] == 'B-ANSWER':
                draw.rectangle(results[i]['bboxes'][j], fill=(127,0,0,127))
            elif results[i]['ner_tags'][j] == 'I-ANSWER':
                draw.rectangle(results[i]['bboxes'][j], fill=(255,0,0,127))
            elif results[i]['ner_tags'][j] == 'O':
                draw.rectangle(results[i]['bboxes'][j], fill=(0,0,0,127))
    image.show()
# for i in range(len(results)):
#         image = dataset["test"][i]["image"]
#         draw = ImageDraw.Draw(image, "RGBA")
#         for r in results[i]:
#             if r.bbox:
#                 if r.type == 'B-HEADER':
#                     draw.rectangle(r.bbox, fill=(0,0,127,127))
#                 elif r.type == 'I-HEADER':
#                     draw.rectangle(r.bbox, fill=(0,0,255,127))
#                 elif r.type == 'B-QUESTION':
#                     draw.rectangle(r.bbox, fill=(0,127,0,127))
#                 elif r.type == 'I-QUESTION':
#                     draw.rectangle(r.bbox, fill=(0,255,0,127))
#                 elif r.type == 'B-ANSWER':
#                     draw.rectangle(r.bbox, fill=(127,0,0,127))
#                 elif r.type == 'I-ANSWER':
#                     draw.rectangle(r.bbox, fill=(255,0,0,127))
#                 elif r.type == 'O':
#                     draw.rectangle(r.bbox, fill=(0,0,0,127))
                    
#         #print("<br>")
#         image.show()
    
