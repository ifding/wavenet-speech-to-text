import argparse
import errno
import os
import time

import torch.distributed as dist
import torch.utils.data.distributed

from tqdm import tqdm
from torch.autograd import Variable

from utils.data import AverageMeter, BucketingSampler 
from utils.data import index2byte as labels
from utils.data_loader import AudioDataLoader, SpectrogramDataset
from utils.decoder import GreedyDecoder
from model.wavenet import WaveNet
from model.deepspeech import DeepSpeech


parser = argparse.ArgumentParser(description='WaveNet training')
parser.add_argument('--data-path', metavar='DIR',
                    help='path to manifest csv', default='data/')
parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--model-path', default='checkpoint/',
                    help='Location to save best validation model')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--layer-size', default=5, type=int, help='10 layer size in paper')
parser.add_argument('--stack-size', default=2, type=int, help='5 stack size in paper')
parser.add_argument('--in-channels', default=20, type=int, help='256 in channels in paper. quantized and one-hot input')
parser.add_argument('--res-channels', default=512, type=int, help='512 res channels in paper')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')


torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)


if __name__ == '__main__':
    args = parser.parse_args() 
    
    train_dataset = SpectrogramDataset(args.data_path,'train')
    valid_dataset = SpectrogramDataset(args.data_path,'valid')   
    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)    
    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
    valid_loader = AudioDataLoader(valid_dataset, 
                                  batch_size=args.batch_size, num_workers=args.num_workers)      

    dtype = torch.FloatTensor
    ltype = torch.LongTensor

    if torch.cuda.is_available():
        print('use gpu')
        dtype = torch.cuda.FloatTensor
        ltype = torch.cuda.LongTensor
    
    model = WaveNet(args.layer_size, args.stack_size, args.in_channels, args.res_channels)
            
    decoder = GreedyDecoder(labels)
    
    avg_loss, start_epoch, start_iter = 0, 0, 0            
    loss_results, cer_results, wer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(
        args.epochs)
    best_wer = None 
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    for epoch in range(start_epoch, args.epochs): 
        end = time.time()
        start_epoch_time = time.time()
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data
            # measure data loading time
            data_time.update(time.time() - end)

            loss_value = model.train(inputs, targets, input_percentages, target_sizes)
            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            if i % 100 == 0:
                print('Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f}' 
                  '({batch_time.avg:.3f})\t Data {data_time.val:.3f}'
                  '({data_time.avg:.3f})\t Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time,
                   data_time=data_time, loss=losses))

        avg_loss /= len(train_sampler)

        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch
        total_cer, total_wer = 0, 0
        with torch.no_grad():
            for i, (data) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                inputs, targets, input_percentages, target_sizes = data

                # unflatten targets
                split_targets = []
                offset = 0
                for size in target_sizes:
                    split_targets.append(targets[offset:offset + size])
                    offset += size

                if torch.cuda.is_available():
                    inputs = inputs.cuda()

                out = model.generate(inputs)  # NxTxH
                seq_length = out.size(1)
                sizes = input_percentages.mul_(int(seq_length)).int()

                decoded_output, _ = decoder.decode(out.data, sizes)
                target_strings = decoder.convert_to_strings(split_targets)
                wer, cer = 0, 0
                for x in range(len(target_strings)):
                    transcript, reference = decoded_output[x][0], target_strings[x][0]
                    wer += decoder.wer(transcript, reference) / float(len(reference.split()))
                    cer += decoder.cer(transcript, reference) / float(len(reference))
                total_cer += cer
                total_wer += wer

            wer = total_wer / len(valid_loader.dataset)
            cer = total_cer / len(valid_loader.dataset)
            wer *= 100
            cer *= 100
            loss_results[epoch] = avg_loss
            wer_results[epoch] = wer
            cer_results[epoch] = cer
            print('Validation Summary Epoch: [{0}]\t Average WER {wer:.3f}\t'
                  'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))
            

        if (best_wer is None or best_wer > wer):
            print("Found better validated model, saving to %s" % args.model_path)
            model.save(args.model_path)
            best_wer = wer

        avg_loss = 0
