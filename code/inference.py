"""inference of the model
"""
import argparse
from transformers import BertForTokenClassification, BertTokenizerFast
from create_dataset import create_dataset
import torch
import result as r

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
    dataset = create_dataset({"train":args.train_file, "test":args.test_file}, label_list)
    tokenized_inputs = []
    SAMPLE_COUNT = 10
    # print(len(dataset["test"]))
    results = {}
    
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
        res = r.Result()
    
        prev_word_id = None
            
        for j, tok_cls in enumerate(predicted_token_class):
            if(word_ids[j] != prev_word_id):
                # new word
                if(res.val != "" and res.type != "O"):
                    results[i] = res
                    # print(res.val, res.type)
                    # res.set_val("")
                    # res.set_type("")
                    # res = r.Result()
                    
                res = r.Result()
                prev_word_id = word_ids[j]
                # print("\n",tokens[i], end='')
                res.set_val(tokens[j])
                res.set_type(tok_cls)
            else:
                # in the word
                if(word_ids[j] != None):
                    # print(tokens[i].replace('#',''),end='')
                    res.append_val(tokens[j].replace('#',''))
        
    # for result in results:
    #     print(result.val, result.type)
    # print(results)
    
    #print the results
    print("Mezery u vstupniho textu nesedi, protoze v datasetu jsou jednotliva slova a dal mezeru po kazdem")
    for i in range(SAMPLE_COUNT):
        print("Vstupni text: ", end='')
        
        for tok in dataset["test"][i]["tokens"]:
            print(tok, end=' ')
        
        print("")
        print("Rozpoznane entity: ")
        if i in results:
            print(results[i].val, results[i].type)
                # print(f"{res.val} - {res.type}")
                
        print("\n")
    
if __name__ == "__main__":
    args = parse_args()
    main(args)