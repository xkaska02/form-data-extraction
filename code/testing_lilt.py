from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification
import evaluate
import numpy as np
from transformers import TrainingArguments
from transformers import Trainer, DataCollatorForTokenClassification


dataset = load_dataset("json",data_files={"data":"data_files/forms_json_dataset/dataset.json"})
train_test_valid = dataset["data"].train_test_split(test_size=0.2, seed=42)
valid_test = train_test_valid["test"].train_test_split(test_size=0.5, seed=42)

dataset = DatasetDict({
    "train": train_test_valid["train"],
    "validation": valid_test["train"],
    "test": valid_test["test"]
})

label_list = ["None","key","information","name","rank","birth_date","nationality","death_date","funeral_date","grave_location","grave_id","information_source","death_book"]
id2label = {id:label for id, label in enumerate(label_list)}
label2id = {label:id for id, label in enumerate(label_list)}


tokenizer = AutoTokenizer.from_pretrained("ufal/robeczech-base", add_prefix_space=True)

def preprocess_function(examples):
    # Tokenize the words
    tokenized_inputs = tokenizer(examples["words"], truncation=True, padding="max_length", is_split_into_words=True)
    
    # Add bounding boxes (you might need to modify this part to fit your data)
    tokenized_inputs["bbox"] = examples["bboxes"]
    tokenized_inputs["labels"] = examples["ner_tags"]
    
    return tokenized_inputs

# Apply preprocessing to your dataset
dataset = dataset.map(preprocess_function, batched=True)

model = AutoModelForTokenClassification.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base", num_labels=len(label_list))

from transformers import Trainer, TrainingArguments

from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (-100)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average="macro")
    return {"precision": precision, "recall": recall, "f1": f1}


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=500,
    save_steps=500,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],  # Assuming you have a train split
    eval_dataset=dataset["test"],  # Assuming you have a test split
    compute_metrics=compute_metrics  # Optional, if you want to compute metrics
)

trainer.train()
