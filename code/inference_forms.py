"""inference of the model
"""
import argparse
from transformers import BertForTokenClassification, BertTokenizerFast
from create_dataset import create_dataset
import torch
from result import Result
from datasets import load_dataset, DatasetDict
from PIL import Image, ImageDraw
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--test_file", default=None)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--validation_file", default=None)
    parser.add_argument("--sample_count", default=None)
    args = parser.parse_args()
    return args

def simple_align(predictions, label_ids, attention_mask):
    preds = np.argmax(predictions, axis=2)[0]  # remove batch dimension
    labels = label_ids[0]
    mask = attention_mask[0]

    preds_aligned = []
    labels_aligned = []

    for pred, label, m in zip(preds, labels, mask):
        if m == 1 and label != -100:
            preds_aligned.append(pred)
            labels_aligned.append(label)

    return preds_aligned, labels_aligned



def main(args):
    # model = BertForTokenClassification.from_pretrained(args.model_path)
    # tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    model_name = "xkaska02/czert_lr2e-05_bs4_train287_max_len8"
    
    model = BertForTokenClassification.from_pretrained(model_name)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    

    
    label_list = ["None","key","information","name","rank","birth_date","nationality","death_date","funeral_date","grave_location","grave_id","information_source","death_book"]
    
    dataset = load_dataset("json", data_files={"train":args.train_file, "validation":args.validation_file, "test":args.test_file})

    
    tokenized_inputs = []
    if args.sample_count:
        SAMPLE_COUNT = int(args.sample_count)
    else:
        SAMPLE_COUNT = len(dataset["test"])
    results : dict[str, Result] = {}
    all_preds = []
    all_labels = []
    
    for i in range(SAMPLE_COUNT):
        inputs = tokenizer(dataset["test"][i]["words"], return_tensors="pt", is_split_into_words=True)
        labels = dataset["test"][i]["ner_tags"]
        word_ids = inputs.word_ids(batch_index=0)
        image_path = dataset["test"][i]["image_name"]
        inputs["labels"] = torch.tensor([[labels[word_id] if word_id is not None else -100 for word_id in word_ids]])
        tokenized_inputs.append(inputs)
        
        tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs[i]["input_ids"][0])

        output = model(tokenized_inputs[i]["input_ids"])
        
        predictions = torch.argmax(output.logits, dim=2)
        y_pred, y_true = simple_align(
            output.logits.detach().cpu().numpy(),
            tokenized_inputs[i]["labels"].cpu().numpy(),
            tokenized_inputs[i]["attention_mask"].cpu().numpy()
        )
        
        all_preds.extend(y_pred)
        all_labels.extend(y_true)
        
        predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
        
        word_ids = tokenized_inputs[i].word_ids(0)
        res = Result()
    
        prev_word_id = None
        
        for j, tok_cls in enumerate(predicted_token_class):
            if(word_ids[j] != prev_word_id):

                if(res.val != ""):
                    res.set_bbox(dataset['test'][i]['bboxes'][word_ids[j-1]])
                    results.setdefault(i, []).append(res)
                    
                res = Result()
                res.set_image_path(image_path)
                prev_word_id = word_ids[j]
                res.set_val(tokens[j])
                res.set_type(tok_cls)
                if word_ids[j]:
                    res.set_bbox(dataset['test'][i]['bboxes'][word_ids[j]])
                
            else:
                # in the word
                if(word_ids[j] != None):
                    res.append_val(tokens[j].replace('#',''))
    
    formatted_results = []
    formatted_result = {"words":[], "bboxes":[], "ner_tags":[], "image_path":""}
    for r in results:
        for word in results[r]:
            formatted_result['words'].append(word.val)
            formatted_result['ner_tags'].append(word.type)
            formatted_result['bboxes'].append(word.bbox)
            formatted_result["image_path"] = word.image_path
            
        formatted_results.append(formatted_result)
        formatted_result = {"words":[], "bboxes":[], "ner_tags":[], "image_path":""} 
    json_object = json.dumps(formatted_results, indent=4)
    # setting output file
    out_file = model_name.split("/")[1]

    with open(f"out/{out_file}.json", "w") as outfile:
        outfile.write(json_object)
    
    print(f"Output data saved into out/{out_file}.json")
        
if __name__ == "__main__":
    args = parse_args()
    main(args)