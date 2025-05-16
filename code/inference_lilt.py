from transformers import AutoModelForTokenClassification
from datasets import load_dataset, DatasetDict
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, LayoutLMv3FeatureExtractor, AutoTokenizer, LayoutLMv3Processor
import torch


def normalize_box(bbox, width, height):
    return [
        int((bbox[0] / width)*1000), # width * (normalized / 1000) = unn --> (unn/width)*1000
        int((bbox[1] / height)*1000),
        int((bbox[2] / width)*1000),
        int((bbox[3] / height)*1000),
    ]

def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]



# model_path = "xkaska02/lilt_xlmroberta_experiment1"
# model_path = "xkaska02/lilt_xlmroberta_small20_trainset_experiment2"
model_path = "xkaska02/lilt_xlmroberta_robeczech_tokenizer_experiment1"



model = AutoModelForTokenClassification.from_pretrained(model_path)

dataset = load_dataset("json", data_files={"train":"data_files/forms_json_dataset/train_split.json", "validation":"data_files/forms_json_dataset/validation_split.json", "test":"data_files/forms_json_dataset/test_split.json"})

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

for i in range(3):
  example = dataset["test"][i]


  image = Image.open(example["image_name"])
  words = example["words"]
  boxes = example["bboxes"]
  word_labels = example["ner_tags"]
  
  for j in range(len(boxes)):
    boxes[j] = normalize_box(boxes[j], image.width, image.height)

  encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")
  
  tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
  

  del encoding["pixel_values"]
  
  with torch.no_grad():
    outputs = model(**encoding)
     
  logits = outputs.logits


  predictions = logits.argmax(-1).squeeze().tolist()

      
  labels = encoding.labels.squeeze().tolist()


  input_ids = encoding["input_ids"].squeeze().tolist()
  tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)

  
  token_boxes = encoding.bbox.squeeze().tolist()
  width, height = image.size

  true_predictions = [model.config.id2label[pred] for pred, label in zip(predictions, labels) if label != - 100]
  true_labels = [model.config.id2label[label] for prediction, label in zip(predictions, labels) if label != -100]
  true_boxes = [unnormalize_box(box, width, height) for box, label in zip(token_boxes, labels) if label != -100]

  from PIL import ImageDraw, ImageFont

  draw = ImageDraw.Draw(image, "RGBA")

  font = ImageFont.load_default()

  def iob_to_label(label):
      if not label:
        return 'other'
      return label

  label2color_fill = {
              "B-key":(255,192,203,127),
            "B-information":(0,0,203,127),
            "B-name":(255,255,0,127),
            "B-rank":(255,0,0,127),
            "B-birth_date":(255,166,0,127),
            "B-nationality":(148,100,10,127),
            "B-death_date":(128,181,31,127),
            "B-funeral_date":(43,130,14,127),
            "B-grave_location":(21,191,186,127),
            "B-grave_id":(8,20,156,127),
            "B-information_source":(108,118,235,127),
            "B-death_book":(184,108,235,127),
  }

  for prediction, box in zip(true_predictions, true_boxes):
      predicted_label = iob_to_label(prediction)

      if predicted_label != "O":
        draw.rectangle(box, fill=label2color_fill[predicted_label])
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color_fill[predicted_label], font=font)
      

  image.show()
  