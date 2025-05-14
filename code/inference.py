"""inference of the model
"""
import argparse
from transformers import BertForTokenClassification, BertTokenizerFast
from create_dataset import create_dataset
import torch
# import result as r
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
    # print(len(dataset["test"]))
    results : dict[str, Result] = {}
    
    for i in range(SAMPLE_COUNT):
        tokenized_inputs.append(tokenizer(dataset["test"][i]["tokens"], return_tensors="pt", is_split_into_words=True))
        # print(tokenized_inputs[i])
        tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs[i]["input_ids"][0])
    # # print(tokens)
        output = model(tokenized_inputs[i]["input_ids"])
    # # print(tokenized_input.word_ids(0))
    # # print(output)
    
    # # print(tokenized_input)
    
    
        predictions = torch.argmax(output.logits, dim=2)
    # # print(predictions)
        predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
    
    
        word_ids = tokenized_inputs[i].word_ids(0)
        res = Result()
    
        prev_word_id = None
            
        for j, tok_cls in enumerate(predicted_token_class):
            if(word_ids[j] != prev_word_id):
                # new word
                # if(res.val != "" and res.type != "O"):
                if(res.val != ""):
                    # results.setdefault(i, {}).setdefault(res.val, res) # create a key if it does not exist and append
                    results.setdefault(i, []).append(res)
                    
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
        # print(results[r])
        for i in range(len(results[r])):
            print(results[r][i].val, end='')
            if i+1 < len(results[r]):
                if(results[r][i+1].type != 'O' and not in_category):
                    # print("dalsi slovo ma kategorii")
                    in_category = True
                    print("<span style='color:blue;'>", end=' ')
                    
                    # print(results[r][i+1].val)
                elif(results[r][i+1].type == 'O' and in_category):
                    # print("konec stejne kategorie slova vypis nazvu kategorie")
                    print(f"</span><sub>{results[r][i].type}</sub>", end=' ')
                    in_category = False
                if(results[r][i+1].val != '.' and results[r][i+1].val != ',' and results[r][i+1].val != '-' and results[r][i].val != '-' and results[r][i+1].val != '?' and results[r][i+1].val != '"' and results[r][i].val != '"' ):
                    print(' ',end='')
        print("<br><br>")
    # for i in range(SAMPLE_COUNT):
    #     # print("Vstupni text: ", end='')
        
    #     for tok in dataset["test"][i]["tokens"]:
    #         if i in results:
    #             if tok in results[i]:
    #                 print(f'<span style="color:blue;">{tok} </span>')
    #                 # print(results[i][tok].type)
    #                 print(f"<sub>{results[i][tok].type}</sub>")
    #             else:
    #                 print(tok, end=' ')            
    #             # if tok in results[i]:
    #             #     print(results[i][tok].type)
        
    #     print("")
    #     # print("Rozpoznane entity: ")
    #     # if i in results:
    #     #     for res in results[i]:
    #     #         print(res.type, res.val)
                
    #     print("")
    #     print("Entity co tam mely byt: ")
    #     print("<ul>")
    #     for j in range(len(dataset["test"][i]["tokens"])):
    #         if(dataset["test"][i]["ner_tags"][j] != 0):
    #             print(f'<li>{model.config.id2label[dataset["test"][i]["ner_tags"][j]]}, {dataset["test"][i]["tokens"][j]}</li>')
    #     print("</ul>")
                
    #     print("\n")
    #print(results)
if __name__ == "__main__":
    args = parse_args()
    main(args)