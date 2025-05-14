"""Create a transformers dataset currently from parquet file
"""

from datasets import load_dataset, Sequence, ClassLabel
import argparse

def parse_args():
    parser = argparse.ArgumentParser(prog="create dataset", description="create dataset from parquet file")
    parser.add_argument("--train_file", default=None, help="file with train data")
    parser.add_argument("--test_file", default=None, help="file with test data")
    args = parser.parse_args()
    return args


def create_dataset(data_files, label_list, file_type, **kwargs):
    if "field" in kwargs:
        dataset = load_dataset(file_type, data_files=data_files, field=kwargs['field'])
    else:
        dataset = load_dataset(file_type, data_files=data_files)
        # this part was here for when i was changing labels in downloaded dataset ##
        new_features = dataset["train"].features.copy()
        
        new_features["ner_tags"] = Sequence(feature=ClassLabel(names=label_list))
        dataset = dataset.cast(new_features)    
    return dataset

# def main(args):
#     data_files = {"train": args.train_file, "test": args.test_file}
#     label_list = ["O", "NUMBER_IN_ADDR", "GEOGRAPHICAL_NAME", "INSTITUTION", "MEDIA", "NUMBER_EXPRESSION", "ARTIFACT_NAME", "PERSONAL_NAME", "TIME_EXPRESSION"]
#     # dataset = create_dataset(data_files=data_files, label_list=label_list)
    # print(dataset["train"][0])
    

# if __name__ == "__main__":
#     args = parse_args()
#     main(args)
