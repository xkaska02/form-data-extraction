"""inference of the model
"""
import argparse
from transformers import BertForTokenClassification, BertTokenizerFast
from create_dataset import create_dataset
import torch
from result import Result
from datasets import load_dataset
from PIL import Image, ImageDraw
import json

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--test_file", default=None)
    parser.add_argument("--train_file", default=None)
    
    args = parser.parse_args()
    return args


def main(args):
    model = BertForTokenClassification.from_pretrained(args.model_path)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    
    
    
    label_list = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
    dataset = load_dataset("nielsr/funsd-iob-original")
    tokenized_inputs = []
    SAMPLE_COUNT = 2
    results : dict[str, Result] = {}
    
    for i in range(SAMPLE_COUNT):
        tokenized_inputs.append(tokenizer(dataset["test"][i]["words"], return_tensors="pt", is_split_into_words=True))
        tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs[i]["input_ids"][0])
        output = model(tokenized_inputs[i]["input_ids"])
    
        predictions = torch.argmax(output.logits, dim=2)

        predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
    
        word_ids = tokenized_inputs[i].word_ids(0)
        res = Result()
    
        prev_word_id = None
        
        for j, tok_cls in enumerate(predicted_token_class):
            if(word_ids[j] != prev_word_id):

                if(res.val != ""):
                    res.set_bbox(dataset['test'][i]['original_bboxes'][word_ids[j-1]])
                    results.setdefault(i, []).append(res)
                    
                res = Result()
                prev_word_id = word_ids[j]
                res.set_val(tokens[j])
                res.set_type(tok_cls)

                if word_ids[j]:
                    res.set_bbox(dataset['test'][i]['original_bboxes'][word_ids[j]])
                
            else:
                if(word_ids[j] != None):
                    res.append_val(tokens[j].replace('#',''))
        
    
    formatted_results = []
    formatted_result = {"words":[], "bboxes":[], "ner_tags":[]}
    for r in results:
        for word in results[r]:
            formatted_result['words'].append(word.val)
            formatted_result['ner_tags'].append(word.type)
            formatted_result['bboxes'].append(word.bbox)
            
        formatted_results.append(formatted_result)
        formatted_result = {"words":[], "bboxes":[], "ner_tags":[]} 
    json_object = json.dumps(formatted_results, indent=4)
    # hardcoded output file
    with open("out/funsd_out.json", "w") as outfile:
        outfile.write(json_object)
    
    print("Output data saved into out/funsd_out.json")
        
if __name__ == "__main__":
    args = parse_args()
    main(args)