# multiple-choice-qa

 This project was intended to familiarize myself with PyTorch DistributedDataParallel (DDP) training with two GPUs on a single node.

## Dataset(s) used
* [SWAG](https://arxiv.org/abs/1808.05326)

## Hyperparameters

* Models were trained on `RoBERTa-BASE`
* Learning Rate was `3e-6` with the AdamW optimizer
* Number of Epochs were `5`

## Results

Results are reported on the model having best valid loss across epochs:

* Accuracy: 83.63 %

Note: Due to compute limitations the models effective batch size of 32 and training time per epoch took 150 minutes. Hence, we can expect much better results on training for more epochs.

 
