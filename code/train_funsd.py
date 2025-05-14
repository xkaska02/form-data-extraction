import argparse
from transformers import BertTokenizerFast
from create_dataset import create_dataset
from transformers import DataCollatorForTokenClassification
import evaluate
import numpy as np
from transformers import BertForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(prog="train", description="train model")

    parser.add_argument("--save_path", default=None, help="")
    parser.add_argument("--model_path", default="UWB-AIR/Czert-B-base-cased", help="Load a model from this path")

    parser.add_argument("--train_file", default=None, help="file with train data")
    parser.add_argument("--test_file", default=None, help="file with test data")
    parser.add_argument("--label_list", nargs="*", default=None, help="list of labels for NER")
    parser.add_argument("--file_type", default=None, help="type of file from which dataset is created")
    parser.add_argument("--train_folder", default=None, help="folder with train files")
    parser.add_argument("--test_folder", default=None, help="folder with test files")

    parser.add_argument("--lr", default=None,help="set learning rate for training", type=float)    
    parser.add_argument("--batch_size", default=None, help="set batch size", type=int)    
    parser.add_argument("--epochs", default=None, help="number of epochs", type=int)
    parser.add_argument("--decay", default=None, help="weight decay", type=float)    
    parser.add_argument("--eval_strat", default=None)
    parser.add_argument("--save_strat", default=None)

    args = parser.parse_args()
    return args


def main(args):
    dataset = load_dataset("nielsr/funsd-iob-original")
    label_list = dataset["train"].features["ner_tags"].feature.names
    
    # removing unnecessary columns
    dataset = dataset.remove_columns(['image','bboxes','original_bboxes'])
    
    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)

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
                label = labels[word_id]
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)
        return new_labels

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["words"], truncation=True, padding=True, is_split_into_words=True)
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
        0: "O", 
        1: "B-HEADER", 
        2: "I-HEADER", 
        3: "B-QUESTION",
        4: "I-QUESTION",
        5: "B-ANSWER",
        6: "I-ANSWER"
    }

    label2id = {
        "O": 0,
        "B-HEADER": 1,
        "I-HEADER": 2,
        "B-QUESTION": 3,
        "I-QUESTION": 4,
        "B-ANSWER": 5,
        "I-ANSWER": 6
        
    }

    model = BertForTokenClassification.from_pretrained(
        args.model_path, num_labels = 7, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir = args.save_path,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.decay,
        evaluation_strategy=args.eval_strat,
        save_strategy=args.save_strat,
        load_best_model_at_end=True,
        optim="adamw_torch"
   )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"], 
        tokenizer=tokenizer,
        data_collator=data_collator, 
        compute_metrics=compute_metrics
    )

    trainer.train()


if __name__=="__main__":
    args=parse_args()
    main(args)