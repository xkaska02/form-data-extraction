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



dataset["train"].to_json("data_files/forms_json_dataset/train_split.json")
dataset["test"].to_json("data_files/forms_json_dataset/test_split.json")
dataset["validation"].to_json("data_files/forms_json_dataset/validation_split.json")
