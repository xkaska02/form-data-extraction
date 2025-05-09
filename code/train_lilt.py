from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import LiltForTokenClassification, LiltModel
import evaluate
import numpy as np
from seqeval.metrics import classification_report
from transformers import TrainingArguments, Trainer
from utils import normalize_box, generate_model_name
from transformers import EarlyStoppingCallback
import argparse
from dotenv import load_dotenv
import os
import wandb
import torch.nn as nn
import json

def parse_args():
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--save_path", default=None, help="")
  parser.add_argument("--model_path", default="nielsr/lilt-xlm-roberta-base", help="path to model (local or from huggingface hub)")
  parser.add_argument("--model_name", default="lilt-xlm-roberta", help="name of the base model that will be used in the output file")
  
  
  parser.add_argument("--train_file", default=None, help="file with train data")
  parser.add_argument("--test_file", default=None, help="file with test data")
  parser.add_argument("--validation_file", default=None, help="file with validation data")
  
  parser.add_argument("--lr", default=None,help="set learning rate for training", type=float)    
  parser.add_argument("--batch_size", default=None, help="set batch size", type=int)    
  parser.add_argument("--epochs", default=None, help="number of epochs", type=int)
  parser.add_argument("--decay", default=None, help="weight decay", type=float)    
  parser.add_argument("--eval_strat", default=None)
  parser.add_argument("--save_strat", default=None)  
  
  parser.add_argument("--classifier_head_layers", choices=["1","2","3"], default=1)
  parser.add_argument("--experiment_name", default="test_project")
  parser.add_argument("--loss_on_subtokens", default=False, type=lambda x: x.lower() == "true", help="if false subtokens will have -100 category and be ignored")
  
  args = parser.parse_args()
  
  return args
    
class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer):
      self.dataset = dataset
      self.tokenizer = tokenizer
    
    def __len__(self):
      return len(self.dataset)
    
    def __getitem__(self, idx):
      example = self.dataset[idx]
      words = example["words"]
      boxes = example["bboxes"]
      ner_tags = example["ner_tags"]
      # image = Image.open(example["image_name"])
      
      # width, height = image.size
      width = 1240
      height = 1744
      bbox = []
      labels = []
      for word, box, label in zip(words, boxes, ner_tags):
        box = normalize_box(box, width, height)
        n_word_tokens = len(self.tokenizer.tokenize(word))
        bbox.extend([box] * n_word_tokens)
        labels.extend([label] + ([-100] * (n_word_tokens - 1)))
      cls_box = sep_box = [0,0,0,0]      
      bbox = [cls_box] + bbox + [sep_box]
      labels = [-100] + labels + [-100]
      
      encoding = self.tokenizer(" ".join(words), truncation=True, max_length=512)
      sequence_length = len(encoding.input_ids)
      # truncate boxes and labels based on length of input ids
      labels = labels[:sequence_length]
      bbox = bbox[:sequence_length]
      
      encoding["bbox"] = bbox
      encoding["labels"] = labels
      
      return encoding

class LiltForTokenClassification2layer(LiltForTokenClassification):
  def __init__(self, config):
    super().__init__(config)
    
    self.num_labels = config.num_labels
    self.lilt = LiltModel(config, add_pooling_layer=False)
    self.dropout - nn.Dropout(config.hidden_dropout_prob)
    
    self.classifier = nn.Sequential(
      nn.Linear(config.hidden_size, config.hidden_size),
      nn.ReLU(),
      nn.Dropout(config.hidden_dropout_prob),
      nn.Linear(config.hidden_size, config.num_labels)
    )
      
    self.post_init()
    
class LiltForTokenClassification3layer(LiltForTokenClassification):
  def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.bert = LiltModel(config, add_pooling_layer=False)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # classifier head
    self.classifier = nn.Sequential(
        nn.Linear(config.hidden_size, config.hidden_size),
        nn.ReLU(),
        nn.Dropout(config.hidden_dropout_prob),
        nn.Linear(config.hidden_size, config.hidden_size),
        nn.ReLU(),
        nn.Dropout(config.hidden_dropout_prob),
        nn.Linear(config.hidden_size, config.num_labels)
    )
      
    self.post_init()

def main(args):
  
  load_dotenv()
  
  api_key=os.getenv("WANDB_API_KEY")
  
  wandb.login(key=api_key)
  os.environ["WANDB_PROJECT"] = args.experiment_name
  
  dataset = load_dataset("json", data_files={"train":args.train_file, "validation":args.validation_file, "test":args.test_file})

  label_list = ["O","B-key","B-information","B-name","B-rank","B-birth_date","B-nationality","B-death_date","B-funeral_date","B-grave_location","B-grave_id","B-information_source","B-death_book"]
  id2label = {id:label for id,label in enumerate(label_list)}
    
  # model_name = "nielsr/lilt-xlm-roberta-base"
  # model_name = "xkaska02/lilt-robeczech-base"
  model_name = args.model_path
  tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True)

  train_dataset = CustomDataset(dataset["train"], tokenizer)
  eval_dataset = CustomDataset(dataset["validation"], tokenizer)
  test_dataset = CustomDataset(dataset["test"], tokenizer)

  # padding_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

  def collate_fn(features):
    boxes = [feature["bbox"] for feature in features]
    labels = [feature["labels"] for feature in features]
    input_ids = [feature["input_ids"] for feature in features]
    
    batch = tokenizer.pad(
      {"input_ids":input_ids}, 
      padding=True,
      # max_length=128,
      return_tensors="pt"
    )
    
    # batch = padding_collator(input_ids)
    
    
    sequence_length = batch["input_ids"].shape[1]
    
    if args.loss_on_subtokens:
      batch["labels"] = torch.tensor([
        label * sequence_length
        for label in labels
      ])
      
      batch["bbox"] = torch.tensor([
        box * sequence_length
        for box in boxes
      ])
    else:
      batch["labels"] = torch.tensor([
        label + [-100] * (sequence_length - len(label))
        for label in labels
      ])
      
      batch["bbox"] = torch.tensor([
        box + [[0,0,0,0]] * (sequence_length - len(box))
        for box in boxes
      ])
    
    return batch

  return_entity_level_metrics = False

  def compute_metrics(p):
      predictions, labels = p
      predictions = np.argmax(predictions, axis=2)

      # Remove ignored index (special tokens)
      true_predictions = [
          [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
          for prediction, label in zip(predictions, labels)
      ]
      true_labels = [
          [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
          for prediction, label in zip(predictions, labels)
      ]

      results = metric.compute(predictions=true_predictions, references=true_labels)
      if return_entity_level_metrics:
          # Unpack nested dictionaries
          final_results = {}
          for key, value in results.items():
              if isinstance(value, dict):
                  for n, v in value.items():
                      final_results[f"{key}_{n}"] = v
              else:
                  final_results[key] = value
          return final_results
      else:
          return {
              "precision": results["overall_precision"],
              "recall": results["overall_recall"],
              "f1": results["overall_f1"],
              "accuracy": results["overall_accuracy"],
          }

  train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
  eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

  if int(args.classifier_head_layers) == 1:
    model = LiltForTokenClassification.from_pretrained(model_name, id2label=id2label)
  elif int(args.classifier_head_layers) == 2:
    model = LiltForTokenClassification2layer.from_pretrained(model_name, id2label=id2label)
  elif int(args.classifier_head_layers) == 3:
    model = LiltForTokenClassification3layer.from_pretrained(model_name, id2label=id2label)
    
  metric = evaluate.load("seqeval")

  if args.save_path:
    save_path = args.save_path
  else:
      save_path = "models"
  output_name = generate_model_name(base=args.model_name, lr=args.lr, batch_size=args.batch_size, train_size=len(dataset["train"]), extra=f"")
  
        
  training_args = TrainingArguments(
        output_dir = f"{save_path}/{output_name}",
        run_name=output_name,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.decay,
        eval_strategy=args.eval_strat,
        save_strategy=args.save_strat,
        load_best_model_at_end=True,
        optim="adamw_torch",
        report_to="wandb",
        metric_for_best_model="eval_loss"
        # eval_accumulation_steps=1
   )
  experiment_name = generate_model_name(base="lilt_xlmroberta", lr=training_args.learning_rate, batch_size=training_args.per_device_train_batch_size, train_size=len(dataset["train"]))

  training_args.output_dir = experiment_name


  early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0001)

  class CustomTrainer(Trainer):
    def get_train_dataloader(self):
      return train_dataloader

    def get_eval_dataloader(self, eval_dataset = None):
      return eval_dataloader

  # test model to cpu idk
  #model.to("cpu")

  # Initialize our Trainer
  trainer = CustomTrainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics,
      callbacks=[early_stopping],
  )

  trainer.train()
  
  test_metrics = trainer.evaluate(eval_dataset=test_dataset)
  with open(f"results/{output_name}.json", "w") as f:
    json.dump(test_metrics, f, indent=2)

  trainer.push_to_hub()
if __name__ == "__main__":
  args = parse_args()
  main(args)