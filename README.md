
# Speech-to-Text using WaveNet


A pytorch implementation of speech recognition based on DeepMind's Paper: [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/pdf/1609.03499.pdf). 

The purpose of this implementation is Well-structured, reusable and easily understandable.

A tensorflow implementation here: [buriburisuri/speech-to-text-wavenet](https://github.com/buriburisuri/speech-to-text-wavenet)

Although WaveNet was designed as a Text-to-Speech model, the paper mentions that they also tested it in the speech recognition task. They didn't give specific details about the implementation, only showed that they achieved 18.8 PER on the test dataset from a model trained directly on raw audio on TIMIT.   

First, I try to use the TIMIT dataset for the speech recognition experiment.

The final architecture is shown in the following figure.

<p align="center">
  <img src="https://raw.githubusercontent.com/ifding/wavenet-speech-to-text/master/log/architecture.png" width="1024"/>
</p>

(Image source: [buriburisuri/speech-to-text-wavenet](https://github.com/buriburisuri/speech-to-text-wavenet))



## Prerequisites

- System
    - Linux 
    - CPU or (NVIDIA GPU + CUDA CuDNN)
    - Python 3.6

- Libraries
    - PyTorch = 0.4.0
    - librosa >= 0.5.1
    

## Setup

- TIMIT Dataset Preprocess

Please prepare TIMIT dataset without modifying the file structure of it and run the following command to preprocess it from wave to MFCC 39 before training.

```
  ./timit_preprocess.sh <TIMIT folder>  
```

After preprocessing step, timit_mfcc_39.pkl should be under your TIMIT folder. Add your data path to config file.


## References

- https://github.com/ibab/tensorflow-wavenet
- https://github.com/vincentherrmann/pytorch-wavenet
- https://github.com/XenderLiu/Listen-Attend-and-Spell-Pytorch