import os
import random
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import AutoTokenizer, LlamaModel, LlamaForSequenceClassification
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType


class CFG:
    NUM_EPOCHS = 1
    BATCH_SIZE = 3
    DROPOUT = 0.05 
    MODEL_NAME = '/mnt/lustre-client/hipo/models/llama-2-7b-chat-hf'
    SEED = 2025 
    MAX_LENGTH = 512 
    NUM_WARMUP_STEPS = 128
    LR_MAX = 5e-5 
    NUM_LABELS = 3 
    LORA_RANK = 3
    LORA_ALPHA = 8
    LORA_MODULES = ['o_proj', 'v_proj']
    
DEVICE = "cuda:0"

def set_seeds(seed):
    """Set seeds for reproducibility """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
 
set_seeds(seed=CFG.SEED)

tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
tokenizer.add_eos_token = True
 
# save tokenizer to load offline during inference
tokenizer.save_pretrained('tokenizer')
def get_token_lengths(texts):
    # tokenize and receive input_ids for reach text
    input_ids = tokenizer(texts.tolist(), return_tensors='np')['input_ids']
    # return length of inputs_ids for each text
    return [len(t) for t in input_ids]
train = pd.read_csv('/home/zzz/dataset/llm-classification-finetuning/train.csv')
def process(input_str):
    stripped_str = input_str.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return  ' '.join(sentences)

 
train.loc[:, 'prompt'] = train['prompt'].apply(process)
train.loc[:, 'response_a'] = train['response_a'].apply(process)
train.loc[:, 'response_b'] = train['response_b'].apply(process)
 
# Drop 'Null' for training
indexes = train[(train.response_a == 'null') & (train.response_b == 'null')].index
train.drop(indexes, inplace=True)
train.reset_index(inplace=True, drop=True)
 
print(f"Total {len(indexes)} Null response rows dropped")
print('Total train samples: ', len(train))
train['text'] = 'User prompt: ' + train['prompt'] +  '\n\nModel A :\n' + train['response_a'] +'\n\n--------\n\nModel B:\n'  + train['response_b']
print(train['text'][4])
train.loc[:, 'token_count'] = get_token_lengths(train['text'])
 
# prepare label for model
train.loc[:, 'label'] = np.argmax(train[['winner_model_a','winner_model_b','winner_tie']].values, axis=1)
 
# Display data
print(train.head())

print(train.label.value_counts())

print('======\nTokenizing ...\n======')
tokens = tokenizer(
    train['text'].tolist(), 
    padding='max_length', 
    max_length=CFG.MAX_LENGTH, 
    truncation=True, 
    return_tensors='np')
 
# Input IDs are the token IDs
INPUT_IDS = tokens['input_ids']
# Attention Masks to Ignore Padding Tokens
ATTENTION_MASKS = tokens['attention_mask']
# Label of Texts
LABELS = train[['winner_model_a','winner_model_b','winner_tie']].values
 
print(f'INPUT_IDS shape: {INPUT_IDS.shape}, ATTENTION_MASKS shape: {ATTENTION_MASKS.shape}')
print(f'LABELS shape: {LABELS.shape}')

print('======\nBuilding dataset and model ...\n======')
def train_dataset(batch_size):
    N_SAMPLES = LABELS.shape[0]
    IDXS = np.arange(N_SAMPLES - (N_SAMPLES % batch_size))
    while True:
        # Shuffle Indices
        np.random.shuffle(IDXS)
        # Iterate Over All Indices Once
        for idxs in IDXS.reshape(-1, batch_size):
            input_ids = torch.tensor(INPUT_IDS[idxs]).to(DEVICE)
            attention_mask = torch.tensor(ATTENTION_MASKS[idxs]).to(DEVICE)
            labels = torch.tensor(LABELS[idxs]).to(DEVICE)  # Multi-label output
            
            
            yield input_ids, attention_mask, labels
 
TRAIN_DATASET = train_dataset(CFG.BATCH_SIZE)
base_model = LlamaForSequenceClassification.from_pretrained(
    CFG.MODEL_NAME,
    num_labels=CFG.NUM_LABELS,
    torch_dtype=torch.bfloat16)
 
base_model.config.pretraining_tp = 1 
 
# Assign Padding TOKEN
base_model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    r=CFG.LORA_RANK,  # the dimension of the low-rank matrices
    lora_alpha = CFG.LORA_ALPHA, # scaling factor for LoRA activations vs pre-trained weight activations
    lora_dropout= CFG.DROPOUT, 
    bias='none',
    inference_mode=False,
    task_type=TaskType.SEQ_CLS,
    target_modules=CFG.LORA_MODULES ) # Only Use Output and Values Projection
model = get_peft_model(base_model, lora_config)
# Trainable Parameters
model.to(DEVICE)
model.print_trainable_parameters()
num_devices = 1
mesh_shape = (1, num_devices, 1)
device_ids = np.array(range(num_devices))
 
print(f'num_devices: {num_devices}')
MODEL_LAYERS_ROWS = []
TRAINABLE_PARAMS = []
N_TRAINABLE_PARAMS = 0
 
for name, param in model.named_parameters():
    # Layer Parameter Count
    n_parameters = int(torch.prod(torch.tensor(param.shape)))
    # Only Trainable Layers
    if param.requires_grad:
        # Add Layer Information
        MODEL_LAYERS_ROWS.append({
            'param': n_parameters,
            'name': name,
            'dtype': param.data.dtype,
        })
        # Append Trainable Parameter
        TRAINABLE_PARAMS.append({ 'params': param })
        # Add Number Of Trainable Parameters"
        N_TRAINABLE_PARAMS += n_parameters
        
print(pd.DataFrame(MODEL_LAYERS_ROWS))
 
print(f"""
===============================
N_TRAINABLE_PARAMS: {N_TRAINABLE_PARAMS:,}
N_TRAINABLE_LAYERS: {len(TRAINABLE_PARAMS)}
===============================
""")

N_SAMPLES = len(train)
STEPS_PER_EPOCH = N_SAMPLES // CFG.BATCH_SIZE
 
OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=CFG.LR_MAX)
 
# Cosine Learning Rate With Warmup
lr_scheduler = transformers.get_cosine_schedule_with_warmup(
    optimizer=OPTIMIZER,
    num_warmup_steps=CFG.NUM_WARMUP_STEPS,
    num_training_steps=STEPS_PER_EPOCH * CFG.NUM_EPOCHS)
 
print(f'BATCH_SIZE: {CFG.BATCH_SIZE}, N_SAMPLES: {N_SAMPLES}, STEPS_PER_EPOCH: {STEPS_PER_EPOCH}')
for state in OPTIMIZER.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and state[k].dtype is not torch.float32:
            state[k] = v.to(dtype=torch.float32)
input_ids, attention_mask, labels = next(TRAIN_DATASET)
 
print(f'input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}')
print(f'attention_mask shape: {attention_mask.shape}, dtype: {attention_mask.dtype}')
print(f'labels shape: {labels.shape}, dtype: {labels.dtype}')


print('\n======\nTraining ...\n======\n')
model.train()
 
# Loss Function, Cross Entropy
LOSS_FN = torch.nn.CrossEntropyLoss().to(dtype=torch.float32)
st = time()

METRICS = {
    'loss': [],
    'accuracy': {'y_true': [], 'y_pred': [] }}
 
for epoch in range(CFG.NUM_EPOCHS):
    ste = time()
    for step in range(STEPS_PER_EPOCH):
        # Zero Out Gradients
        OPTIMIZER.zero_grad()
        
        # Get Batch
        input_ids, attention_mask, labels = next(TRAIN_DATASET)
        
        # Forward Pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
       
        # Logits Float32
        logits = outputs.logits.to(dtype=torch.float32)
        
        # Backward Pass
        loss = LOSS_FN(logits, labels.to(dtype=torch.float32))
        loss.backward()
        
        # optimizer step
        OPTIMIZER.step()
        
        # Update Learning Rate Scheduler
        lr_scheduler.step()
        
        # Update Metrics And Progress Bar
        METRICS['loss'].append(float(loss))
        METRICS['accuracy']['y_true'] += labels.squeeze().tolist()
        METRICS['accuracy']['y_pred'] += torch.argmax(F.softmax(logits, dim=-1), dim=1).cpu().tolist()
        
        if (step + 1) % 50 == 0:  
            metrics = 'µ_loss: {:.3f}'.format(np.mean(METRICS['loss']))
            metrics += ', step_loss: {:.3f}'.format(METRICS['loss'][-1])
            metrics += ', µ_auc: {:.3f}'.format(accuracy_score(torch.argmax(torch.tensor(METRICS['accuracy']['y_true']).view(-1, CFG.NUM_LABELS), dim=-1), \
                                                               METRICS['accuracy']['y_pred']))
            lr = OPTIMIZER.param_groups[0]['lr']
            print(f'{epoch+1:02}/{CFG.NUM_EPOCHS:02} | {step+1:04}/{STEPS_PER_EPOCH} lr: {lr:.2E}, {metrics}', end='')
            print(f'\nSteps per epoch: {step+1} complete | Time elapsed: {time()- st}')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }
            torch.save(dict([(k,v) for k, v in model.named_parameters() if v.requires_grad]), 'model_epoch_{epoch}.pth')
            # torch.save(checkpoint, f'./checkpoints/model_epoch_{epoch}.pt')
            print(f'Model saved at epoch {epoch}| Elapsed time: {time() - st} ')
    
    print(f'\nEpoch {epoch+1} Completed | Total time for epoch: {time() - ste} ' )