
# Speech-to-Text using WaveNet


**Still need to figure out CTCLoss nan problem**


A pytorch implementation of speech recognition based on DeepMind's Paper: [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/pdf/1609.03499.pdf). 

The purpose of this implementation is Well-structured, reusable and easily understandable.

A tensorflow implementation here: [buriburisuri/speech-to-text-wavenet](https://github.com/buriburisuri/speech-to-text-wavenet)

Although WaveNet was designed as a Text-to-Speech model, the paper mentions that they also tested it in the speech recognition task. They didn't give specific details about the implementation, only showed that they achieved 18.8 PER on the test dataset from a model trained directly on raw audio on TIMIT.   

I modify the WaveNet model from [https://github.com/golbin/WaveNet](https://github.com/golbin/WaveNet) and apply the [PyTorch bindings for Warp-ctc](https://github.com/SeanNaren/warp-ctc) for the speech recognition experiment.

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
    - librosa = 0.5.0
    - https://github.com/SeanNaren/warp-ctc
    - pandas >= 0.19.2
    - [scikits.audiolab](https://pypi.python.org/pypi/scikits.audiolab)==0.11.0


## Dataset

We used [VCTK](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html), 
[LibriSpeech](http://www.openslr.org/12/) and [TEDLIUM release 2](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus) corpus.
Total number of sentences in the training set composed of the above three corpus is 240,612. 
Valid and test set is built using only LibriSpeech and TEDLIUM corpuse, because VCTK corpus does not have valid and test set. 
After downloading the each corpus, extract them in the 'asset/data/VCTK-Corpus', 'asset/data/LibriSpeech' and 
 'asset/data/TEDLIUM_release2' directories. 

The TEDLIUM release 2 dataset provides audio data in the SPH format, so we should convert them to some format 
librosa library can handle. Run the following command in the 'asset/data' directory convert SPH to wave format.  
<pre><code>
find -type f -name '*.sph' | awk '{printf "sox -t sph %s -b 16 -t wav %s\n", $0, $0".wav" }' | bash
</code></pre>

If you don't have installed `sox`, please installed it first.
<pre><code>
sudo apt-get install sox
</code></pre>

We found the main bottle neck is disk read time when training, so we decide to pre-process the whole audio data into 
  the MFCC feature files which is much smaller. And we highly recommend using SSD instead of hard drive.  
  Run the following command in the console to pre-process whole dataset.
<pre><code>
python preprocess.py
</code></pre>


## Training 

```
python train.py
```


## References

- https://github.com/ibab/tensorflow-wavenet
- https://github.com/vincentherrmann/pytorch-wavenet
- https://github.com/XenderLiu/Listen-Attend-and-Spell-Pytorch
- https://github.com/SeanNaren/deepspeech.pytorch
- https://github.com/golbin/WaveNet