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

def id_to_name(label_idx):
    json_path ="ontology/ontology.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
    mapdata = {item["id"]: item["name"] for item in data}
    result = [mapdata[id] for id in label_idx if id in mapdata]
    if 'Silence'  in result:
        result.remove('Silence')
    if 'Music' in result:
        result.remove('Music')
    if 'Speech' in result:
        result.remove('Speech')
    if 'Background music' in result:
        result.remove('Background music') 
    return result
# predict the classification probability of each class

# Function to append data to a JSON file
def append_to_json_file(json_file, entries):
    try:
        # Load existing data from the JSON file
        with open(json_file, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        # If file does not exist, start with an empty list
        data = []

    # Append new data
    data.extend(entries)

    # Write the updated data back to the JSON file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

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
