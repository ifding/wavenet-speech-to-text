# reference : https://github.com/Faur/TIMIT
# 			  https://github.com/jameslyons/python_speech_features/issues/53
import os
import sys
import wave
import timeit; program_start_time = timeit.default_timer()
import random; random.seed(int(timeit.default_timer()))
from six.moves import cPickle 
import numpy as np
import scipy.io.wavfile as wav

import python_speech_features as features
# a python package for speech features at https://github.com/jameslyons/python_speech_features

if len(sys.argv) != 3:
	print('Usage: python3 preprocess.py <timit directory> <output_file>')


##### SCRIPT META VARIABLES #####
phn_file_postfix = '.phn'
RIFF_wav_postfix = '.WAV'

##### Validation split #####
# default using 5% of data as validation
val_split 	= 0.05 


data_type = 'float32'

paths = sys.argv[1]
train_source_path	= os.path.join(paths, 'train')
test_source_path	= os.path.join(paths, 'test')
target_path			= os.path.join(paths, sys.argv[2])


# 61 different phonemes
phonemes = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh", 
	"f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "w", "y", 
	"hh", "hv", "el", "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy",
	"ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]

phonemes2index = {k:v for v,k in enumerate(phonemes)}


def get_total_duration(file):
	"""Get the length of the phoneme file, i.e. the 'time stamp' of the last phoneme"""
	for line in reversed(list(open(file))):
		[_, val, _] = line.split()
		return int(val)


def create_mfcc(filename):
	"""Perform standard preprocessing, as described by Alex Graves (2012)
	http://www.cs.toronto.edu/~graves/preprint.pdf
	Output consists of 12 MFCC and 1 energy, as well as the first derivative of these.
	[1 energy, 12 MFCC, 1 diff(energy), 12 diff(MFCC)
	"""

	(rate,sample) = wav.read(filename)

	mfcc = features.mfcc(sample, rate, winlen=0.025, winstep=0.01, numcep = 13, nfilt=26,
	preemph=0.97, appendEnergy=True)
	d_mfcc = features.delta(mfcc, 2)
	a_mfcc = features.delta(d_mfcc, 2)

	out = np.concatenate([mfcc, d_mfcc, a_mfcc], axis=1)

	return out, out.shape[0]

def calc_norm_param(X):
	"""Assumes X to be a list of arrays (of differing sizes)"""
	total_len = 0
	mean_val = np.zeros(X[0].shape[1])
	std_val = np.zeros(X[0].shape[1])
	for obs in X:
		obs_len = obs.shape[0]
		mean_val += np.mean(obs,axis=0)*obs_len
		std_val += np.std(obs, axis=0)*obs_len
		total_len += obs_len
	
	mean_val /= total_len
	std_val /= total_len

	return mean_val, std_val, total_len

def normalize(X, mean_val, std_val):
	for i in range(len(X)):
		X[i] = (X[i] - mean_val)/std_val
	return X

def set_type(X, type):
	for i in range(len(X)):
		X[i] = X[i].astype(type)
	return X


def preprocess_dataset(source_path):
	"""Preprocess data, ignoring compressed files and files starting with 'SA'"""
	i = 0
	X = []
	Y = []

	for dirName, subdirList, fileList in os.walk(source_path):
		for fname in fileList:
			if not fname.endswith(phn_file_postfix):
				continue

			phn_fname = dirName + '/' + fname
			wav_fname = dirName + '/' + fname[0:-4] + RIFF_wav_postfix

			total_duration = get_total_duration(phn_fname)
			fr = open(phn_fname)

			X_val, total_frames = create_mfcc(wav_fname)
			total_frames = int(total_frames)

			X.append(X_val)

			y_val = np.zeros(total_frames) - 1
			start_ind = 0
			for line in fr:
				[start_time, end_time, phoneme] = line.rstrip('\n').split()
				start_time = int(start_time)
				end_time = int(end_time)

				phoneme_num = phonemes2index[phoneme] if phoneme in phonemes2index else -1
				end_ind = int(np.round((end_time)/total_duration*total_frames))
				y_val[start_ind:end_ind] = phoneme_num

				start_ind = end_ind
			fr.close()

			if -1 in y_val:
				print('WARNING: -1 detected in TARGET')
				print(y_val)

			Y.append(y_val.astype('int32'))

			i+=1
			print('file No.',i, end='\r', flush=True)

	print('Done')
	return X, Y


##### PREPROCESSING #####
print()

print('Preprocessing training data...')
X_train_all, y_train_all = preprocess_dataset(train_source_path)
print('Preprocessing testing data...')
X_test, y_test = preprocess_dataset(test_source_path)
print('Preprocessing completed.')

print()
train_size 	= len(X_train_all)
print('Collected {} training instances from {} (should be 4620 in complete TIMIT )'.format(train_size,train_source_path))
print('Collected {} testing instances from {} (should be 1680 in complete TIMIT )'.format(len(X_test),test_source_path))

print('Spliting {} out of {} ({}%) training data as validation set.'.format(int(train_size*val_split),train_size,val_split*100))
val_idx = [int(i) for i in random.sample(range(0, train_size), int(train_size*val_split))]

X_train = []; X_val = []
y_train = []; y_val = []
for i in range(len(X_train_all)):
    if i in val_idx:
        X_val.append(X_train_all[i])
        y_val.append(y_train_all[i])
    else:
        X_train.append(X_train_all[i])
        y_train.append(y_train_all[i])

print()
print('Normalizing data to let mean=0, sd=1 for each channel.')

mean_val, std_val, _ = calc_norm_param(X_train)

X_train = normalize(X_train, mean_val, std_val)
X_val 	= normalize(X_val, mean_val, std_val)
X_test 	= normalize(X_test, mean_val, std_val)

X_train = set_type(X_train, data_type)
X_val 	= set_type(X_val, data_type)
X_test 	= set_type(X_test, data_type)

print()
print('Saving data to ',target_path)
with open(target_path + '.pkl', 'wb') as cPickle_file:
    cPickle.dump(
        [X_train, y_train, X_val, y_val, X_test, y_test], 
        cPickle_file, 
        protocol=cPickle.HIGHEST_PROTOCOL)

print()
print('Preprocessing completed in {:.3f} secs.'.format(timeit.default_timer() - program_start_time))
