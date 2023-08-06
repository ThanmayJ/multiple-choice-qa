# multiple-choice-qa

 This project was intended to familiarize myself with PyTorch DistributedDataParallel (DDP) and FullyShardedDataParallel (FSDP) training. I've tested it with two GPUs on a single node.

## Task
Multiple Choice Question Answering: Given a question, classify the answer from a given set of possible answers.

## Dataset(s) used
* [SWAG](https://arxiv.org/abs/1808.05326)

## Hyperparameters

* Models were trained on `RoBERTa-BASE`
* Learning Rate was `3e-6` (using RoBERTaMultipleChoice) and `1e-5` (using RoBERTaSequenceClassification) with the AdamW optimizer
* Number of Epochs were `5`

## Results

Results are reported on the model having best valid loss across epochs:

* Accuracy using RoBERTaSequenceClassification: 85.60 %
* Accuracy using RoBERTaMultipleChoice: 83.63 %

Note: Due to compute limitations the models effective batch size of 32 and training time per epoch took 150 minutes. Hence, we can expect much better results on training for more epochs.

## References
* [Advanced Model Training with Fully Sharded Data Parallel - PyTorch](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html)
* [Getting Started with Fully Sharded Data Parallel - PyTorch](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
* [A Comprehensive Tutorial to Pytorch DistributedDataParallel - Medium](https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51)
* [Getting Started with Distributed Data Parallel - PyTorch](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
* [https://pytorch.org/docs/stable/fsdp.html](https://pytorch.org/docs/stable/fsdp.html)
* [https://pytorch.org/docs/stable/distributed.html](https://pytorch.org/docs/stable/distributed.html)
* [Two is Better than Many? Binary Classification as an Effective Approach to Multi-Choice Question Answering](https://arxiv.org/abs/2210.16495)

 
