"""inference of the model
"""
import argparse
# from transformers import BertForTokenClassification, BertTokenizerFast
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3TokenizerFast, LayoutLMv3Processor
from create_dataset import create_dataset
import torch
# import result as r
from result import Result
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import json

def unnormalize_box(bbox, width, height):
            return [
                width * (bbox[0] / 1000),
                height * (bbox[1] / 1000),
                width * (bbox[2] / 1000),
                height * (bbox[3] / 1000),
            ]

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--test_file", default=None)
    parser.add_argument("--train_file", default=None)
    
    args = parser.parse_args()
    return args


def main(args):
    model = LayoutLMv3ForTokenClassification.from_pretrained(args.model_path)
    
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    
    dataset = load_dataset("nielsr/funsd-layoutlmv3")
    
    sample_count = 3
    
    
    for i in range(sample_count):        
        example = dataset["test"][i]
    
        image = example["image"]
        words = example["tokens"]
        boxes = example["bboxes"]
        word_labels = example["ner_tags"]
        

        encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**encoding)
            
        logits = outputs.logits
        
        predictions = logits.argmax(-1).squeeze().tolist()
        labels = encoding.labels.squeeze().tolist()
        
        

        token_boxes = encoding.bbox.squeeze().tolist()
        width, height = image.size

        true_predictions = [model.config.id2label[pred] for pred, label in zip(predictions, labels) if label != - 100]
        true_labels = [model.config.id2label[label] for prediction, label in zip(predictions, labels) if label != -100]
        true_boxes = [unnormalize_box(box, width, height) for box, label in zip(token_boxes, labels) if label != -100]
        
        draw = ImageDraw.Draw(image,"RGBA")
        font = ImageFont.load_default()
        
        def iob_to_label(label):
            label = label[2:]
            if not label:
                return "other"
            return label
            
        label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}

        for prediction, box in zip(true_predictions, true_boxes):
            predicted_label = iob_to_label(prediction).lower()
            draw.rectangle(box, outline=label2color[predicted_label])
            draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)

        image.show()
                    
if __name__ == "__main__":
    args = parse_args()
    main(args)