"""parses funsd annotation into txt that will be used to train BERT"""

from datasets import load_dataset
from transformers import BertTokenizerFast

data_folder = "data_files/FUNSD/dataset"

dataset = load_dataset(
    "json",
    data_files={
        "train": f"{data_folder}/training_data/annotations/*.json",
        "test": f"{data_folder}/testing_data/annotations/*.json"
        },
    field="form"
    )

print(dataset['train'][0])

tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")