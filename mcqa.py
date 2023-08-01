import os
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import evaluate
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoModelForSequenceClassification

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

MODEL_NAME = "roberta-base"
MODEL_PATH = f"models/{MODEL_NAME} "
TOKENIZER_PATH = f"tokenizers/{MODEL_NAME}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

device = 'cuda'

swag = load_dataset("swag", "regular")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class MultipleChoiceDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent1 = self.data[idx]["sent1"]
        sent2 = self.data[idx]["sent2"]
        label = self.data[idx]["label"]
        sentences1 = [sent1]*4
        sentences2 = []
        for i in range(4):
            sentences2.append(sent2+" "+self.data[idx][f"ending{i}"])
        source = self.tokenizer(sentences1, sentences2, max_length=self.max_len, pad_to_max_length=True,
                                truncation=True, padding="max_length", return_tensors='pt')  # shape [4,max_len]

        return {'source_ids': source.input_ids.to(dtype=torch.long),
                'source_masks': source.attention_mask.to(dtype=torch.long),
                'label': label}
    
class SequenceClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = {'sentence':[], 'label':[]}
        for idx in range(len(data)):
            sent1 = data[idx]["sent1"]
            sent2 = data[idx]["sent2"]
            label = data[idx]["label"]
            sentences = [sent1]*4
            labels = [0,0,0,0]
            labels[label] = 1
            for i in range(4):
                endingi = f'ending{i}'
                sentences[i] = f"{sentences[i]} {sent2} {data[idx][endingi]}"
            self.data['sentence'].extend(sentences)
            self.data['label'].extend(labels)
    
    def __len__(self):
        return len(self.data['label'])
    
    def __getitem__(self, idx):
        sentence = self.data['sentence'][idx]
        label = self.data['label'][idx]
        source = self.tokenizer(sentence, max_length=self.max_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt') # shape [max_len]
        
        return {'source_ids':source.input_ids.squeeze(0).to(dtype=torch.long),
                'source_masks':source.attention_mask.squeeze(0).to(dtype=torch.long),
                'label':label}


def get_ddp_loader(rank, world_size, batch_size, model_type):
    if model_type == "MultipleChoice":
        SWAGDataset = MultipleChoiceDataset
    else:
        SWAGDataset = SequenceClassificationDataset
    
    train_set = SWAGDataset(swag["train"].select(range(100)), tokenizer)
    valid_set = SWAGDataset(swag["validation"].select(range(10)), tokenizer)
    test_set = SWAGDataset(swag["test"].select(range(10)), tokenizer)
    # train_set = SWAGDataset(swag["train"], tokenizer)
    # valid_set = SWAGDataset(swag["validation"], tokenizer)
    # test_set = SWAGDataset(swag["test"], tokenizer)
    
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    valid_sampler = DistributedSampler(valid_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=False, sampler=train_sampler)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=False, sampler=valid_sampler)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=False, sampler=test_sampler)
    
    dataloader = {"train":train_loader, "valid":valid_loader, "test":test_loader}
    return dataloader

TEAMLoss = torch.nn.CrossEntropyLoss()

def train(rank, epoch, tokenizer, model, device, loader, optimizer, model_type):
    model.train()
    epoch_loss = 0
    loader.sampler.set_epoch(epoch)    
    for _, data in enumerate(loader):
        optimizer.zero_grad()
        src = data["source_ids"].to(rank)  # shape (BATCH_SIZE, 4, MAX_LEN) or (BATCH_SIZE, MAX_LEN)
        mask = data["source_masks"].to(rank)   # shape (BATCH_SIZE, 4, MAX_LEN) or (BATCH_SIZE, MAX_LEN)
        label = data["label"].to(rank)  # shape (BATCH_SIZE)

        if model_type=="MultipleChoice":
            outputs = model(input_ids=src, attention_mask=mask, labels=label)
            loss = outputs[0]
        else:
            outputs = model(input_ids=src,attention_mask=mask)
            loss = TEAMLoss(outputs.logits, label)

        if _ % 10 == 0 and rank==0:
            print(f"Epoch {epoch} | Step {_} | Loss {loss}")

        loss.backward()
        optimizer.step()
        epoch_loss += loss
    return epoch_loss/len(loader)


def validate(rank, epoch, tokenizer, model, device, loader, model_type):
    model.eval()
    epoch_loss = 0
    loader.sampler.set_epoch(epoch)    
    with torch.no_grad():
        for _, data in enumerate(loader):
            src = data["source_ids"].to(rank)    # shape (BATCH_SIZE, 4, MAX_LEN) or (BATCH_SIZE, MAX_LEN)
            mask = data["source_masks"].to(rank)     # shape (BATCH_SIZE, 4, MAX_LEN) or (BATCH_SIZE, MAX_LEN)
            label = data["label"].to(rank)  # shape (BATCH_SIZE)

            outputs = model(input_ids=src, attention_mask=mask, labels=label)

            if model_type=="MultipleChoice":
                outputs = model(input_ids=src, attention_mask=mask, labels=label)
                loss = outputs[0]
            else:
                outputs = model(input_ids=src,attention_mask=mask)
                loss = TEAMLoss(outputs.logits, label)

            if _ % 10 == 0 and rank==0:
                print(f"Epoch {epoch} | Step {_} | Loss {loss}")

            epoch_loss += loss
    return epoch_loss/len(loader)


def do_gather(rank, world_size, data):
    if rank == 0:
        # create an empty list we will use to hold the gathered values
        output = [torch.zeros_like(data) for _ in range(world_size)]
        dist.gather(tensor=data, gather_list=output, group=dist.group.WORLD)
        return output
    else:
        dist.gather(tensor=data, gather_list=[], group=dist.group.WORLD)
        return []


def test(rank, world_size, epoch, tokenizer, model, device, loader, model_type):
    model.eval()
    epoch_loss = 0
    references = []
    predictions = []
    loader.sampler.set_epoch(epoch)    
    with torch.no_grad():
        for _, data in enumerate(loader):
            src = data["source_ids"].to(rank)  # shape (BATCH_SIZE, MAX_LEN) or (BATCH_SIZE, MAX_LEN)
            mask = data["source_masks"].to(rank)        # shape (BATCH_SIZE, MAX_LEN) or (BATCH_SIZE, MAX_LEN)
            label = data["label"].to(rank)  # shape (BATCH_SIZE)

            outputs = model(input_ids=src, attention_mask=mask, labels=label)

            loss = outputs[0]
            logits = outputs[1]

            if model_type=="MultipleChoice":
                outputs = model(input_ids=src, attention_mask=mask, labels=label)
                loss = outputs[0]
                logits = outputs[1]
                pred_label = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
            else:
                outputs = model(input_ids=src,attention_mask=mask)
                logits = outputs.logits
                loss = TEAMLoss(logits, label)

                positive_score = logits[:, 1]
                print(positive_score.shape)
                pred_label = torch.argmax(positive_score.reshape(-1, 4),dim=-1)
                label =  label.reshape(-1,4).argmax(1)

            predictions.extend(pred_label.tolist())
            references.extend(label.tolist())

            if _ % 10 == 0 and rank==0:
                print(f"Epoch {epoch} | Step {_} | Loss {loss}")

            epoch_loss += loss
    
    # print(f"Rank {rank} has {len(predictions)} predictions and {len(references)} references")
    predictions = torch.tensor(predictions,dtype=torch.int8).to(rank)
    references = torch.tensor(references,dtype=torch.int8).to(rank)
    dist.barrier()
    torch.cuda.empty_cache()
    # gather_predictions = do_gather(rank, world_size, predictions)
    # gather_references = do_gather(rank, world_size, references)
    gather_predictions = [torch.zeros(len(predictions),dtype=torch.int8).to(rank) for _ in range(world_size)]
    gather_references = [torch.zeros(len(references),dtype=torch.int8).to(rank) for _ in range(world_size)]
    if rank==0:
        dist.gather(tensor=predictions, gather_list=gather_predictions, dst=0)
        dist.gather(tensor=references, gather_list=gather_references, dst=0)
    else:
        dist.gather(tensor=predictions, gather_list=[], dst=0)
        dist.gather(tensor=references, gather_list=[], dst=0)
    dist.barrier() # Waits till all processes obtain model outputs before evaluating
    
    if(rank==0):
        print("Gathered all predictions and references. Evaluating...")
    
    # Evaluation
    predictions_list = [x.item() for y in gather_predictions for x in y]
    references_list = [x.item() for y in gather_references for x in y]
    evaluate_accuracy = evaluate.load("accuracy")
    accuracy = evaluate_accuracy.compute(references=references_list, predictions=predictions_list)
    # if(rank==0):
    #     print(f"After gathering, we have in total {len(predictions_list)} predictions and {len(references_list)} references")

    return accuracy, epoch_loss/len(loader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main(rank, args):
    world_size = args.gpus
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    #####
    dataloader = get_ddp_loader(rank, world_size, args.batch_size, args.model_type)
    if args.model_type=="MultipleChoice":
        model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME).to(rank)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,num_labels=2).to(rank)
    
    model = DDP(model, device_ids=[rank], output_device=rank)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    
    print(f"Rank {rank} is using GPU: {torch.cuda.get_device_name(rank)}")
    
    NUM_EPOCHS = args.num_epochs
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    train_start = time.time()
    for epoch in range(0, NUM_EPOCHS, 1):
        start_time = time.time()
        if(rank==0):
            print("\nTraining...")
        train_loss = train(rank, epoch, tokenizer, model, device, dataloader["train"], optimizer, args.model_type)
        end_time = time.time()
        if(rank==0):
            print("\nValidating...")
        valid_loss = validate(rank, epoch, tokenizer, model, device, dataloader["valid"], args.model_type)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if rank==0:
            print(f"[Epoch {epoch}] Time: {epoch_mins}m {epoch_secs}s | Train Loss {train_loss} | Valid Loss {valid_loss} \n")
        
        if valid_loss < best_valid_loss and rank==0:
            print(f"[Saving Model]")
            checkpoint_dict = {'model':model.state_dict(), 'optimizer':optimizer.state_dict()}
            torch.save(checkpoint_dict, MODEL_PATH)
            # model.module.save_pretrained(MODEL_PATH)
            tokenizer.save_pretrained(TOKENIZER_PATH)
            best_valid_loss = valid_loss

    train_end = time.time()
    train_mins, train_secs = epoch_time(train_start, train_end)
    if rank==0:
        print("\nTesting...")
        print(f"[Loading Model with Best Validation Loss]")
    dist.barrier() # Better to do this before loading the model to avoid PytorchStreamReader errors
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint_dict = torch.load(MODEL_PATH, map_location=map_location)
    model.load_state_dict(checkpoint_dict['model'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    accuracy, test_loss = test(rank, world_size, epoch, tokenizer, model, device, dataloader["valid"], args.model_type)
    if rank==0:
        print(f"Accuracy: {accuracy} | Test Loss: {test_loss}")
        print(f"Training time took {train_mins}m {train_secs}s")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-a', '--ipaddr', default='localhost', type=str, 
                        help='IP address of the main node')
    parser.add_argument('-p', '--port', default='12355', type=str, 
                        help='Port main node')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='batch size for train, valid and test splits')
    parser.add_argument('--num_epochs', default=5, type=int, 
                        help='Number of training epochs')
    parser.add_argument('--lr', default=1e-5, type=int, 
                        help='Learning rate for AdamW optimizer')
    parser.add_argument('--model_type', default="MultipleChoice",type=str,
                        help='Type of MCQA model: MultipleChoice or SequenceClassification')
    
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.ipaddr
    os.environ['MASTER_PORT'] = args.port
    
    mp.spawn(main, args=(args,), nprocs=args.gpus)
