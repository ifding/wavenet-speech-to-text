import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.data import index2byte as labels


       
class SpectrogramDataset(Dataset):
    def __init__(self, _data_path, set_name='train'):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:
        1580-141084-0015.flac,9,6,1,10,14,17,19,6,20,20,6,5,1,...
        """
        # load meta file
        mfcc_files, label = [], []
        with open(_data_path + 'preprocess/meta/%s.csv' % set_name) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                # mfcc file
                mfcc_files.append(_data_path + 'preprocess/mfcc/' + row[0] + '.npy')
                # label info ( convert to string object for variable-length support )
                label.append(np.asarray(row[1:], dtype=np.int).tolist())
                
        self.mfcc_files = mfcc_files
        self.label = label
        self.size = len(mfcc_files)
        super(SpectrogramDataset, self).__init__()

    def __getitem__(self, index):
        audio_path = self.mfcc_files[index]
        # load mfcc
        mfcc = np.load(audio_path, allow_pickle=False)
        # speed perturbation augmenting
        spect = self.augment_speech(mfcc)
        spect = torch.FloatTensor(spect)
        # normalize
        mean = spect.mean()
        std = spect.std()
        spect.add_(-mean)
        spect.div_(std)        
        transcript = self.label[index]       
        return spect, transcript

    def augment_speech(self, mfcc):
        # random frequency shift ( == speed perturbation effect on MFCC )
        r = np.random.randint(-2, 2)
        # shifting mfcc
        mfcc = np.roll(mfcc, r, axis=0)
        # zero padding
        if r > 0:
            mfcc[:r, :] = 0
        elif r < 0:
            mfcc[r:, :] = 0
        return mfcc
    
    def __len__(self):
        return self.size
    

def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)    
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn    

        

        