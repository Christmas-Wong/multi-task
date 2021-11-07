'''
Author: your name
Date: 2021-10-16 22:21:50
LastEditTime: 2021-11-07 22:05:33
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /multi_task_demo/src/test.
'''

from datasets import load_dataset
from model.semh_model import BertForSequenceClassification
from trainer.trainer_semh import MultiTaskTrainer
from entity.data_collator import NLPDataCollator
from utils.util_model_eval import multitask_eval
from utils.util_model_save import save_model
from utils.util_common import generate_feature
from utils.util_read_file import get_label_encoder
from entity.arg_model import ModelArguments, DataTrainingArguments
from transformers import (
    BertTokenizer,
    TrainingArguments,
    HfArgumentParser
    )
from args.arg_common import (
    TASK_DATA_FILE,
    TASK_NUM_CLASS,
    COLUMUNS_DICT,
    MAX_LENGTH,
    TASK_LE,
    AVERAGE_DICT
    )


if __name__ == '__main__': 
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    dataset_dict = {
        
        "location": load_dataset('csv', data_files={'train': TASK_DATA_FILE["location"]["train"], 
                                                 'test': TASK_DATA_FILE["location"]["test"]}),
        "service": load_dataset('csv', data_files={'train': TASK_DATA_FILE["service"]["train"], 
                                                    'test': TASK_DATA_FILE["service"]["test"]})
    }
    
    multitask_model = BertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        task_labels_map={"location": TASK_NUM_CLASS["location"], "service": TASK_NUM_CLASS["service"]},
    )
    tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path, truncation=True)
    
    label_encoder_dict = {
        "location": get_label_encoder(TASK_LE["location"]),
        "service": get_label_encoder(TASK_LE["service"])
    }
    features_dict = generate_feature(label_encoder_dict,
                                    tokenizer,
                                    MAX_LENGTH,
                                    dataset_dict,
                                    COLUMUNS_DICT)
    train_dataset = {
        task_name: dataset["train"] for task_name, dataset in features_dict.items()
    }
    
    trainer = MultiTaskTrainer(
        model=multitask_model,
        args=training_args,
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset
        )
    trainer.train()
    save_model(tokenizer, multitask_model, data_args.data_save_dir)
    multitask_eval(multitask_model,tokenizer, dataset_dict,label_encoder_dict, AVERAGE_DICT)

