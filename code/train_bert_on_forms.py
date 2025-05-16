import argparse
from transformers import BertTokenizerFast, RobertaTokenizerFast
from create_dataset import create_dataset
from transformers import DataCollatorForTokenClassification
import evaluate
import numpy as np
from transformers import BertForTokenClassification, TrainingArguments, Trainer, RobertaForTokenClassification
from datasets import load_dataset, DatasetDict
from dotenv import load_dotenv
import os
import wandb
from utils import generate_model_name
from transformers import EarlyStoppingCallback
import json
from transformers import BertPreTrainedModel, BertModel, RobertaModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn as nn
import torch

def parse_args():
    parser = argparse.ArgumentParser(prog="train", description="train model")

    parser.add_argument("--save_path", default=None, help="")
    parser.add_argument("--model_path", default="UWB-AIR/Czert-B-base-cased", help="Load a model from this path")
    parser.add_argument("--model_name", default="czert", help="name of the base model that will be used in the output file")
    
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
    
    parser.add_argument("--model_type", default="bert", help="set model type")
    
    args = parser.parse_args()
    return args


class BertForTokenClassification2layer(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        
        self.post_init()
        
        # self.forward = BertForTokenClassification.forward

class RobertaForTokenClassification2layer(RobertaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels=config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        
        self.post_init()

class BertForTokenClassification3layer(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
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

class RobertaForTokenClassification3layer(RobertaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
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
    
    # setting up wandb tracking
    api_key=os.getenv("WANDB_API_KEY")

    wandb.login(key=api_key)
    
    os.environ["WANDB_PROJECT"] = args.experiment_name

    dataset = load_dataset("json", data_files={"train":args.train_file, "validation":args.validation_file, "test":args.test_file})
    
    

    label_list = ["O","B-key","B-information","B-name","B-rank","B-birth_date","B-nationality","B-death_date","B-funeral_date","B-grave_location","B-grave_id","B-information_source","B-death_book"]
    
    if args.model_type == "bert":
        tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    elif args.model_type == "roberta":
        tokenizer = RobertaTokenizerFast.from_pretrained(args.model_path, add_prefix_space=True)
    

    def align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                new_labels.append(-100)
            else:
                if args.loss_on_subtokens:
                    new_labels.append(labels[word_id])
                else:
                    new_labels.append(-100)
            
        return new_labels
    max_len = 256
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["words"], truncation=True, padding=True, is_split_into_words=True, max_length=max_len)
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels, in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels,word_ids))
            
        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    seqeval = evaluate.load("seqeval")
    
    labels = [label_list[i] for i in dataset["train"][0]["ner_tags"]]
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    id2label = {
            0:"O",
            1:"B-key",
            2:"B-information",
            3:"B-name",
            4:"B-rank",
            5:"B-birth_date",
            6:"B-nationality",
            7:"B-death_date",
            8:"B-funeral_date",
            9:"B-grave_location",
            10:"B-grave_id",
            11:"B-information_source",
            12:"B-death_book",
    }

    label2id = {
            "O":0,
            "B-key":1,
            "B-information":2,
            "B-name":3,
            "B-rank":4,
            "B-birth_date":5,
            "B-nationality": 6,
            "B-death_date": 7,
            "B-funeral_date":8,
            "B-grave_location":9,
            "B-grave_id":10,
            "B-information_source":11,
            "B-death_book":12,
    }
    
    model_type = args.model_type
    
    if model_type == "bert":
        if int(args.classifier_head_layers) == 1:
            model = BertForTokenClassification.from_pretrained(
                args.model_path, num_labels = 13, id2label=id2label, label2id=label2id
            )
        elif int(args.classifier_head_layers) == 2:
            model = BertForTokenClassification2layer.from_pretrained(
                args.model_path, num_labels = 13, id2label=id2label, label2id=label2id
            )
        elif int(args.classifier_head_layers) == 3:
            model = BertForTokenClassification3layer.from_pretrained(
                args.model_path, num_labels = 13, id2label=id2label, label2id=label2id
            )
    elif model_type == "roberta":
        if int(args.classifier_head_layers) == 1:
            model = RobertaForTokenClassification.from_pretrained(
                args.model_path, num_labels = 13, id2label=id2label, label2id=label2id
            )
        elif int(args.classifier_head_layers) == 2:
            model = RobertaForTokenClassification2layer.from_pretrained(
                args.model_path, num_labels = 13, id2label=id2label, label2id=label2id
            )
        elif int(args.classifier_head_layers) == 3:
            model = RobertaForTokenClassification3layer.from_pretrained(
                args.model_path, num_labels = 13, id2label=id2label, label2id=label2id
            )
    
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
   )
    
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0001)
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"], 
        tokenizer=tokenizer,
        data_collator=data_collator, 
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )

    trainer.train()

    test_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    with open(f"results/{output_name}.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    trainer.push_to_hub()
if __name__=="__main__":
    args=parse_args()
    main(args)