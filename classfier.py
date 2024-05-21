import torch
from beats.BEATs import BEATs, BEATsConfig
from torchaudio import transforms as T
import json
from tqdm import tqdm
import numpy as np
from data.dataload import AudioDataset,collate_fn
from torch.utils.data import DataLoader

# load the fine-tuned checkpoints
checkpoint = torch.load('BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')

cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()

BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BEATs_model.to(device)


data_set = AudioDataset()
val_dataloder=DataLoader(data_set,
        batch_size=500,
        num_workers=64,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
        )

# Define the path to the JSON file
with torch.no_grad():
    entries = []
    all_probs = []
    for batch in tqdm(val_dataloder):
        waveform, wav_path = batch
        probs = BEATs_model.extract_features(waveform)[0]
        all_probs.append(probs)

stacked_probs = torch.vstack(all_probs)
np.save('probs.npy', stacked_probs.cpu().numpy())
