from datasets import load_dataset

# dataset = load_dataset("json", data_files={"train":"data_files/forms_json_dataset/train_split.json", "validation":"data_files/forms_json_dataset/validation_split.json", "test":"data_files/forms_json_dataset/test_split.json"})
dataset = load_dataset("json", data_files="data_files/forms_json_dataset/train_split150.json")
print(dataset)
exit(0)

values_seen = {i: 0 for i in range(13)}

keys_to_check = [k for k in values_seen.keys() if k not in (1,2)]
for data_entry in dataset["train"]:

    for tag in data_entry["ner_tags"]:
        # print(tag)
        values_seen[tag]+=1
    
print(values_seen)