'''
Author: your name
Date: 2021-11-07 23:00:28
LastEditTime: 2021-11-07 23:30:04
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /server_multitask/src/inference.py
'''
from utils.util_time import calculate_time
from typing import Dict, List
from transformers import BertTokenizer
from model.semh_model import BertForSequenceClassification
from args.arg_common import TASK_LE, TASK_NUM_CLASS
from utils.util_read_file import get_label_encoder

@calculate_time(3)
def inference(text: str,
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
                                max_length= 512,
                                padding = 'max_length',
                                return_token_type_ids= False,
                                return_attention_mask= True,
                                truncation=True,
                                return_tensors = 'pt'
                            ).to("cpu")
            label_code = model.predict(**inputs, task_name=["useful", "department"])
            result = {"washed_text": text}
            for key, value in label_code.items():
                predictions_label = label_encoder_dict[key].inverse_transform([value])
                result = {
                            key+"_label": predictions_label[0],
                            key+"_label_code": value
                        }

            return  result


def main():
    token_path = "/mnt/nfsshare/wangfei/project/server_multitask/model/"
    model_path = "/mnt/nfsshare/wangfei/project/server_multitask/model/"
    text_test = ["服务员没有征询我们的意见，做成几分熟，我的同事觉得她的那份熟过头了，吃起来有点硬了，口感不是很好",
                "地方很好找，就在万达广场一五楼，出了直梯左转就是，停车也很方便，带小子来的，还提供了儿童安全座椅，服务很贴心"]

    result = []
    tokenizer = BertTokenizer.from_pretrained(token_path, truncation=True)
    multitask_model = BertForSequenceClassification.from_pretrained(
        model_path,
        task_labels_map={"location": TASK_NUM_CLASS["location"], "service": TASK_NUM_CLASS["service"]},
    )

    label_encoder_dict = {
        "location": get_label_encoder(TASK_LE["location"]),
        "service": get_label_encoder(TASK_LE["service"])
    }

    for sentence in text_test:
        text = sentence.strip()
        # Data Wash
        # inference
        dict_ele = inference(text, tokenizer, multitask_model, label_encoder_dict)
        dict_ele["origin_text"] = sentence
        result.append(dict_ele)
    # return result
    print(result)


if __name__ == "__main__":
    main()