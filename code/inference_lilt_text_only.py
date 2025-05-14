from transformers import AutoProcessor, AutoModelForTokenClassification, LayoutLMv3FeatureExtractor, XLMRobertaTokenizerFast, LayoutLMv3Processor, AutoTokenizer, RobertaTokenizerFast
from PIL import Image
import torch
from datasets import load_dataset
from result import Result
import json
from utils import normalize_box
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--test_file", default=None)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--validation_file", default=None)
    parser.add_argument("--sample_count", default=None)
    args = parser.parse_args()
    return args

def main(args):
    
    model_id = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)
    # tokenizer = RobertaTokenizerFast.from_pretrained(model_id, add_prefix_space=True)
    # feature_extractor = LayoutLMv3FeatureExtractor().from_pretrained("microsoft/layoutlmv3-base")


    model = AutoModelForTokenClassification.from_pretrained(model_id)

    dataset = load_dataset("json", data_files={"train":"data_files/forms_json_dataset/train_split.json", "validation":"data_files/forms_json_dataset/validation_split.json", "test":"data_files/forms_json_dataset/test_split.json"})

    # print(dataset["test"][0])
    if args.sample_count:
        SAMPLE_COUNT = int(args.sample_count)
    else:
        SAMPLE_COUNT = len(dataset["test"])
    
    formatted_results = []

    for i in range(SAMPLE_COUNT):
        
        image = Image.open(dataset["test"][i]["image_name"])
        words = dataset["test"][i]["words"]
        boxes = dataset["test"][i]["bboxes"]
        image_path = dataset["test"][i]["image_name"]


        normalized_boxes = [normalize_box(b, image.width, image.height) for b in boxes]
        # image_features = feature_extractor(images=image, return_tensors="pt")

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        encoding['bbox'] = torch.tensor([normalized_boxes + [[0, 0, 0, 0]] * (512 - len(normalized_boxes))])  # pad to 512
        # encoding['image'] = image_features['pixel_values']
        del encoding['offset_mapping']

        # print(tokenizer.convert_ids_to_tokens(encoding["input_ids"][0]))
        # exit(0)


        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)


        predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
        # print(predicted_token_class)
        # exit(0)
        tokenized_inputs = []
        tokenized_inputs.append(tokenizer(words, return_tensors="pt", is_split_into_words=True, padding="max_length",max_length=512))
        tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs[0]["input_ids"][0])
        # print(len(tokenized_words[0]["input_ids"][0]))
        # print(tokenized_words[0].word_ids(0))
        # print(tokens)


        word_ids = tokenized_inputs[0].word_ids(0)
        prev_word_id = None
        results = {}
        res = Result()

        for j, tok_cls in enumerate(predicted_token_class):
                    if(word_ids[j] != prev_word_id):
                        # new word
                        # if(res.val != "" and res.type != "O"):

                        if(res.val != ""):
                            # results.setdefault(i, {}).setdefault(res.val, res) # create a key if it does not exist and append
                            # print("appenduju tohle", vars(res))
                            # print(f"appenduju v j {j}")
                            res.set_bbox(dataset['test'][i]['bboxes'][word_ids[j-1]])
                            results.setdefault(i, []).append(res)
                            
                        res = Result()
                        res.set_image_path(image_path)
                        prev_word_id = word_ids[j]
                        res.set_val(tokens[j].replace("▁","")) #! this symbol '▁'is not an underscore
                        res.set_type(tok_cls)
                        # print(word_ids[j])
                        if word_ids[j]:
                            # print(f"{word_ids[j]} {dataset['test'][i]['original_bboxes'][word_ids[j]]}")
                            # print(f"j: {j}, word_ids[j]: {word_ids[j]}, j-1: {j-1}, word_ids[j-1]: {word_ids[j-1]}")
                            res.set_bbox(dataset['test'][i]['bboxes'][word_ids[j]])
                            # print(f"res.bbox {res.bbox}")
                        
                    else:
                        # in the word
                        if(word_ids[j] != None):
                            res.append_val(tokens[j].replace('▁',''))
                            
        # print(results)
        formatted_result = {"words":[], "bboxes":[], "ner_tags":[], "image_path":""}
        for r in results:
            for word in results[r]:
                # formatted_results[r]
                # print(word.__dict__)
                formatted_result['words'].append(word.val)
                formatted_result['ner_tags'].append(word.type)
                formatted_result['bboxes'].append(word.bbox)
                formatted_result["image_path"] = word.image_path
                
            formatted_results.append(formatted_result)
            formatted_result = {"words":[], "bboxes":[], "ner_tags":[], "image_path":""}
        # print(formatted_results)

        json_object = json.dumps(formatted_results, indent=4)


    model_name = model_id.split("/")[1]
    out_path = f"out/{model_name}.json"
    # print(out_path)

    with open(out_path, "w") as outfile:
        outfile.write(json_object)
        
    print(f"Output data saved into {out_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)