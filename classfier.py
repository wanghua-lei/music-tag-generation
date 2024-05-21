import torch
from beats.BEATs import BEATs, BEATsConfig
from torchaudio import transforms as T
from tqdm import tqdm
import numpy as np
from data.dataload import AudioDataset,collate_fn
from torch.utils.data import DataLoader
from accelerate import Accelerator

# load the fine-tuned checkpoints
checkpoint = torch.load('BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')

cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()


data_set = AudioDataset()
val_dataloder=DataLoader(data_set,
        batch_size=200,
        num_workers=2,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
        )

# Use Accelerator
accelerator = Accelerator()
BEATs_model, val_dataloder = accelerator.prepare(BEATs_model, val_dataloder)

# Use DataParallel
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# BEATs_model.to(device)
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs")
#     BEATs_model = torch.nn.DataParallel(BEATs_model)

# if isinstance(BEATs_model,torch.nn.DataParallel):
BEATs_model = BEATs_model.module

# Define the path to the JSON file
with torch.no_grad():
    entries = []
    all_probs = []
    for batch in tqdm(val_dataloder):
        waveform, wav_path = batch
        probs = BEATs_model.extract_features(waveform)[0]
        probs = accelerator.gather_for_metrics(probs)
        all_probs.append(probs)

stacked_probs = torch.vstack(all_probs)
np.save('probs.npy', stacked_probs.cpu().numpy())
