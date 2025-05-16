"""inference of the model
"""
import argparse
from transformers import BertForTokenClassification, BertTokenizerFast
from create_dataset import create_dataset
import torch
from result import Result

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
    
    
    
    label_list = ["O", "NUMBER_IN_ADDR", "GEOGRAPHICAL_NAME", "INSTITUTION", "MEDIA", "NUMBER_EXPRESSION", "ARTIFACT_NAME", "PERSONAL_NAME", "TIME_EXPRESSION"]
    dataset = create_dataset({"train":args.train_file, "test":args.test_file}, label_list, file_type="parquet")
    tokenized_inputs = []
    SAMPLE_COUNT = 10
    results : dict[str, Result] = {}
    
    for i in range(SAMPLE_COUNT):
        tokenized_inputs.append(tokenizer(dataset["test"][i]["tokens"], return_tensors="pt", is_split_into_words=True))
        tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs[i]["input_ids"][0])
        output = model(tokenized_inputs[i]["input_ids"])
    
    
        predictions = torch.argmax(output.logits, dim=2)

        predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
    
    
        word_ids = tokenized_inputs[i].word_ids(0)
        res = Result()
    
        prev_word_id = None
            
        for j, tok_cls in enumerate(predicted_token_class):
            if(word_ids[j] != prev_word_id):
                # new word
                # if(res.val != "" and res.type != "O"):
                if(res.val != ""):
                    results.setdefault(i, []).append(res) # create a key if it does not exist and append
                    
                res = Result()
                prev_word_id = word_ids[j]
                res.set_val(tokens[j])
                res.set_type(tok_cls)
            else:
                # in the word
                if(word_ids[j] != None):
                    res.append_val(tokens[j].replace('#',''))
        
    
    #print the results
    in_category = False
    for r in results:
        for i in range(len(results[r])):
            print(results[r][i].val, end='')
            if i+1 < len(results[r]):
                if(results[r][i+1].type != 'O' and not in_category):
                    in_category = True
                    print("<span style='color:blue;'>", end=' ')
                elif(results[r][i+1].type == 'O' and in_category):
                    print(f"</span><sub>{results[r][i].type}</sub>", end=' ')
                    in_category = False
                if(results[r][i+1].val != '.' and results[r][i+1].val != ',' and results[r][i+1].val != '-' and results[r][i].val != '-' and results[r][i+1].val != '?' and results[r][i+1].val != '"' and results[r][i].val != '"' ):
                    print(' ',end='')
        print("<br><br>")

if __name__ == "__main__":
    args = parse_args()
    main(args)