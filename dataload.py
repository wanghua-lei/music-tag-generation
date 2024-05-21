import json
import random
import numpy
import torch
from torch.utils.data import Dataset
from pedalboard.io import AudioFile
import torchaudio
from torchaudio import transforms as T
import torch.nn.functional as F

class AudioDataset(Dataset):

    def __init__(self,):
        super(AudioDataset, self).__init__()

        # self.sr = 16000
        # self.max_length = 10 * self.sr
        json_path = "json_files/raw_30s_cleantags.json" #
        with open(json_path, 'r') as f:
            json_obj = json.load(f)

        self.wav_paths = [item["location"] for item in json_obj]

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index):
        audio_name = self.wav_paths[index].split("/")[-1]

        # caption = self.captions[index]
        wav_path = self.wav_paths[index]

        #load
        with AudioFile(wav_path) as f:
            waveform = f.read(f.frames)
            waveform = torch.from_numpy(waveform)
            sr = f.samplerate
        if waveform.shape[0] > 1:
            waveform = (waveform[0] + waveform[1]) / 2
        else:
            waveform = waveform.squeeze(0)
        
        if sr != 16000:
            resample_tf = T.Resample(sr, 16000)
            waveform = resample_tf(waveform)

        max_length = 160000
        if waveform.shape[-1] > max_length:
            max_start = waveform.shape[-1] - max_length
            start = random.randint(0, max_start)
            waveform = waveform[start: start + max_length]
        else:
            waveform=F.pad(waveform, (0, max_length - waveform.shape[-1]), "constant", 0.0)

        return waveform, wav_path

def collate_fn(batch):
    wav_list = []
    wav_path_list = []
    for waveform, pad, text, path in batch:
        wav_list.append(waveform)
        wav_path_list.append(path)
    waveforms = torch.stack(wav_list, dim=0)
    # duration = Tensor(duration_list).type(torch.long)
    return waveforms, wav_path_list

