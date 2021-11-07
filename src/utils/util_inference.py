'''
Author: your name
Date: 2021-10-29 11:03:24
LastEditTime: 2021-10-29 11:04:54
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /server_mdap/src/utils/util_inference.py
'''

from typing import Dict
from utils.util_time import calculate_time
from transformers import BertTokenizer
from entity.model.semh_model import BertForSequenceClassification
import torch
from args.arg_common import MAX_LENGTH

@calculate_time(3)
def inference(text: str,
                task_name: str,
                tokenizer: BertTokenizer,
                model: BertForSequenceClassification,
                label_encoder_dict: Dict
            
            ) -> Dict:
            if not text or len(text)<1:
                return  {
                            "washed_text": "",
                            "label": None,
                            "label_code": None
                        }
            inputs = tokenizer.encode_plus(
                                text,
                                None,
                                add_special_tokens=True,
                                max_length= MAX_LENGTH,
                                padding = 'max_length',
                                return_token_type_ids= False,
                                return_attention_mask= True,
                                truncation=True,
                                return_tensors = 'pt'
                            ).to("cpu")
            # inputs = inputs.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
            # inputs = features_dict[task_name]["test"][index]["input_ids"]
            # print(inputs)
            # labels = features_dict[task_name]["test"][index]["labels"].item()

            logits = model(**inputs, task_name=task_name)[0]

            predictions_code = torch.argmax(
                torch.FloatTensor(torch.softmax(logits, dim=1).detach().cuda().tolist()),
                dim=1,
            )
            predictions_label = label_encoder_dict[task_name].inverse_transform([predictions_code])

            result = {
                "washed_text": text,
                "label": predictions_label[0],
                "label_code": predictions_code.item()
            }

            return  result
