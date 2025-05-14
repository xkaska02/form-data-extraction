from datasets import load_dataset
from PIL import Image, ImageDraw


dataset = load_dataset("json", data_files={"train":"data_files/forms_json_dataset/train_split.json", "validation":"data_files/forms_json_dataset/validation_split.json", "test":"data_files/forms_json_dataset/test_split.json"})

label2color_fill = [
            (255,192,203,127),
            (0,0,203,127),
            (255,255,0,127),
            (255,0,0,127),
            (255,166,0,127),
            (148,100,10,127),
            (128,181,31,127),
            (43,130,14,127),
            (21,191,186,127),
            (8,20,156,127),
            (108,118,235,127),
            (184,108,235,127),
]

SAMPLE_SIZE = 4
for i in range(SAMPLE_SIZE):

    img = Image.open(dataset["test"][i]["image_name"])

    draw = ImageDraw.Draw(img, "RGBA")
    # print(len(dataset["test"][i]["ner_tags"]))
    
    for j in range(len(dataset["test"][i]["ner_tags"])):
        
        box = dataset["test"][i]["bboxes"][j]
        predicted_label = dataset["test"][i]["ner_tags"][j]
        # print(predicted_label)
        if(predicted_label):
            draw.rectangle(box, fill=label2color_fill[predicted_label-1])


    img.show()