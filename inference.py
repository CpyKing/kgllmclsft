# 为什么训练没有爆显存，推理爆显存
import torch
import sklearn
import numpy as np
import pandas as pd
import time
 
from transformers import AutoTokenizer, LlamaModel, LlamaForSequenceClassification, BitsAndBytesConfig
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from torch.cuda.amp import autocast
from threading import Thread
 
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
 
if (not torch.cuda.is_available()): print("Sorry - GPU required!")
MODEL_NAME = '/mnt/lustre-client/hipo/models/llama-2-7b-chat-hf'
WEIGHTS_PATH = './checkpoints/model_epoch_0.pt'
MAX_LENGTH = 512
BATCH_SIZE = 1
DEVICE = torch.device("cuda:0")    
test = pd.read_csv('/home/zzz/dataset/llm-classification-finetuning/test.csv')
sample_sub = pd.read_csv('/home/zzz/dataset/llm-classification-finetuning/sample_submission.csv')
 
# concatenate strings in list
def process(input_str):
    stripped_str = input_str.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return  ' '.join(sentences)
 
test.loc[:, 'prompt'] = test['prompt'].apply(process)
test.loc[:, 'response_a'] = test['response_a'].apply(process)
test.loc[:, 'response_b'] = test['response_b'].apply(process)
test['text'] = 'User prompt: ' + test['prompt'] +  '\n\nModel A :\n' + test['response_a'] +'\n\n--------\n\nModel B:\n'  + test['response_b']
print(test['text'][0]) 

tokenizer = AutoTokenizer.from_pretrained('/home/zzz/01/kaggle_llm_classification_finetuning/tokenizer')
tokens = tokenizer(test['text'].tolist(), padding='max_length',
                   max_length=MAX_LENGTH, truncation=True, return_tensors='pt')
 
INPUT_IDS = tokens['input_ids'].to(DEVICE, dtype=torch.int32)
ATTENTION_MASKS = tokens['attention_mask'].to(DEVICE, dtype=torch.int32)
 
# Move tensors to CPU and convert them to lists
input_ids_cpu = [tensor.cpu().tolist() for tensor in INPUT_IDS]
attention_masks_cpu = [tensor.cpu().tolist() for tensor in ATTENTION_MASKS]
 
data = pd.DataFrame()
data['INPUT_IDS'] = input_ids_cpu
data['ATTENTION_MASKS'] = attention_masks_cpu
data[:2]
bnb_config =  BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=False)
 
# Load base model on GPU 0
device0 = torch.device('cuda:0')
 
base_model_0 = LlamaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map='cuda:0')
# base_model_0 = LlamaForSequenceClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=3,
#     torch_dtype=torch.bfloat16)
base_model_0.config.pad_token_id = tokenizer.pad_token_id
 
peft_config = LoraConfig(
    r=3,
    lora_alpha=8,
    lora_dropout=0.05,
    bias='none',
    inference_mode=True,
    task_type=TaskType.SEQ_CLS,
    target_modules=['o_proj', 'v_proj'])
# Get peft
model_0 = get_peft_model(base_model_0, peft_config).to(device0) 
# Load weights
model_0.load_state_dict(torch.load(WEIGHTS_PATH)['model_state_dict'], strict=False)
model_0.eval()
 
model_0.print_trainable_parameters()
import gc
gc.collect()
def inference(df, model, device, batch_size=BATCH_SIZE):
    input_ids = torch.tensor(df['INPUT_IDS'].values.tolist(), dtype=torch.long)
    attention_mask = torch.tensor(df['ATTENTION_MASKS'].values.tolist(), dtype=torch.long)
    
    generated_class_a = []
    generated_class_b = []
    generated_class_c = []
 
    model.eval()
    
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        batch_input_ids = input_ids[start_idx:end_idx].to(device)
        batch_attention_mask = attention_mask[start_idx:end_idx].to(device)
        
        with torch.no_grad():
            with autocast():
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
        
        probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        
        generated_class_a.extend(probabilities[:, 0])
        generated_class_b.extend(probabilities[:, 1])
        generated_class_c.extend(probabilities[:, 2])
    
    df['winner_model_a'] = generated_class_a
    df['winner_model_b'] = generated_class_b
    df['winner_tie'] = generated_class_c
 
    torch.cuda.empty_cache()  
 
    return df
st = time.time()
 
N_SAMPLES = len(data)

 
# Function to run inference in a thread
def run_inference(df, model, device, results, index):
    results[index] = inference(df, model, device)
 
# Dictionary to store results from threads
results = {}

run_inference(data, model_0, device0, results, 0)

# Combine results back into the original DataFrame
data = results[0]
 
print(f"Processing complete. Total time: {time.time() - st}")
TARGETS = ['winner_model_a', 'winner_model_b', 'winner_tie']
 
sample_sub[TARGETS] = data[TARGETS]
print(sample_sub)
sample_sub.to_csv('submission.csv', index=False)