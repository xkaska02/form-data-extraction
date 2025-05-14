import json
from datasets import DatasetDict, load_dataset

dataset = load_dataset("json",data_files={"data":"data_files/forms_json_dataset/dataset_without_keys.json"})
train_test_valid = dataset["data"].train_test_split(test_size=0.3, seed=42)

valid_test = train_test_valid["train"].train_test_split(test_size=0.2, seed=42)
    
dataset = DatasetDict({
    "train": valid_test["train"],
    "validation": valid_test["test"],
    "test": train_test_valid["test"]
})



# with open("data_files/forms_json_dataset/dataset_int_bboxes.json","r") as file:
#     data = json.load(file)


# remove labels for keys only leave labels for values    
# new_list = []
# for d in data:
#     # print(d)
#     new_tags = [tag if tag != 1 else 0 for tag in d["ner_tags"]]
#     d["ner_tags"] = new_tags
    
# print(data[0]["ner_tags"])
# one_form_list = []

# example = data[0]
# for i in range(500):
#     example["id"] = i
#     one_form_list.append(example)
    
# # print(one_form_list)
dataset["train"].to_json("data_files/forms_json_dataset/train_split.json")
dataset["test"].to_json("data_files/forms_json_dataset/test_split.json")
dataset["validation"].to_json("data_files/forms_json_dataset/validation_split.json")

# json_object = json.dumps(dataset["train"])

# print(json_object)

# with open("data_files/forms_json_dataset/dataset_without_keys.json","w") as outfile:
#     outfile.write(json_object)