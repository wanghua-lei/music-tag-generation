import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import re
from typing import Dict, List

import csv
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import pathlib
import librosa
import lightning.pytorch as pl
from models.clap_encoder import CLAP_Encoder
import json
from data.WavCaps_dataset import AudioTextDataset
from data.datamodules import DataModule


sys.path.append('../AudioSep/')
from utils import (
    load_ss_model,
    # calculate_sdr,
    # calculate_sisdr,
    parse_yaml,
    get_mean_sdr_from_dict,
)

def calculate_sdr_list(
    refs: list,
    ests: list,
    eps=1e-10
) -> list:
    r"""Calculate SDRs for lists of reference and estimated signals.

    Args:
        refs (list): List of reference signals (each a np.ndarray)
        ests (list): List of estimated signals (each a np.ndarray)
        eps (float): Small positive constant to avoid division by zero

    Returns:
        list: List of SDR values
    """
    sdrs = []
    for ref, est in zip(refs, ests):
        reference = np.array(ref)
        noise = np.array(est) - reference

        numerator = np.clip(a=np.mean(reference ** 2), a_min=eps, a_max=None)
        denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)

        sdr = 10. * np.log10(numerator / denominator)
        sdrs.append(sdr)

    return sdrs


def calculate_sisdr_list(ref, est):
    r"""Calculate SDR between reference and estimation.

    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """

    eps = np.finfo(ref.dtype).eps

    sisdrs = []
    for ref, est in zip(ref, est):
        reference = ref.copy()
        estimate = est.copy()
        reference = reference.reshape(reference.size, 1)
        estimate = estimate.reshape(estimate.size, 1)

        Rss = np.dot(reference.T, reference)
        # get the scaling factor for clean sources
        a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)

        e_true = a * reference
        e_res = estimate - e_true

        Sss = (e_true**2).sum()
        Snn = (e_res**2).sum()

        sisdr = 10 * np.log10((eps+ Sss)/(eps + Snn))
        sisdrs.append(sisdr)

    return sisdrs

class WavCapsEvaluator:
    def __init__(
        self,
        sampling_rate=32000,
        datatype="",
    ) -> None:
        r"""Clotho evaluator.
        Returns:
            None
        """
        # with open('evaluation/metadata/clotho_eval.csv') as csv_file:
        #     csv_reader = csv.reader(csv_file, delimiter=',')
        #     eval_list = [row for row in csv_reader][1:]

        # with open("data_Wavcaps/json_files/AudioSet_SL/as.json") as f:
        #     acdata = json.load(f)
        # result =[]
        # for item in acdata:
        #     result.append([item["id"],item["caption"],item["duration"]])
        # self.eval_list = result

        self.sampling_rate = sampling_rate

        datamodule = DataModule(
            num_workers=8,
            batch_size=8,
            datatype=datatype
        )
        self.loader = datamodule.train_dataloader()

        if datatype=="AS":
            self.save_clean = "cleaned/ac.json"
        elif datatype=="BBC":
            self.save_clean = "cleaned/bbc.json"
        elif datatype=="FSD":
            self.save_clean = "cleaned/fsd.json"
        elif datatype=="SB":
            self.save_clean = "cleaned/sb.json"

    def __call__(
        self,
        pl_model: pl.LightningModule
    ) -> Dict:
        r"""Evalute."""

        pl_model.eval()
        device = pl_model.device

        sisdrs_list = []
        sdrs_list = []

        result=[]
        with torch.no_grad():
            with open(self.save_clean, 'w') as json_file:
                for batch_data in tqdm(self.loader, total=len(self.loader)):

                    audioname, source, text, duration = batch_data
                    # source_path = os.path.join(self.audio_dir, audioname)  #raw audio

                    # mixture_path = os.path.join(self.audio_dir, f'mixture-{idx}.wav') #混合
                    # mixture, fs = librosa.load(mixture_path, sr=self.sampling_rate, mono=True)
                    # sdr_no_sep = calculate_sdr(ref=source, est=mixture)   
                    # text = [caption]

                    conditions = pl_model.query_encoder.get_query_embed(
                        modality='text',
                        text=text,
                        device=device 
                    )

                    input_dict = {
                        "mixture": torch.Tensor(source).to(device),#[None, None, :]
                        "condition": conditions,
                    }

                    sep_segment = pl_model.ss_model(input_dict)["waveform"]
                    # sep_segment: (batch_size=1, channels_num=1, segment_samples)

                    sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()
                    # sep_segment: (segment_samples,)

                    sdr = calculate_sdr_list(refs=source.numpy(), ests=sep_segment)

                    # sdri = sdr - sdr_no_sep
                    sisdr = calculate_sisdr_list(ref=source.numpy(), est=sep_segment)

                    mean_sisdr = np.mean(sisdr)
                    mean_sdri = np.mean(sdr)
                    print("Audio start:{}, Avg SDR: {:.3f},Avg SISDR: {:.3f}".format(audioname[0].split('/')[-1],mean_sdri, mean_sisdr))
                    # sisdrs_list.extend(sisdr)
                    # sdrs_list.extend(sdr)

                    for i, sdr_i in enumerate(sdr):
                        if sdr_i >10:
                            cleaned_map = {
                                'idx': audioname[i],
                                'caption': text[i],
                                'duration': duration[i]
                            }
                            json_file.write(json.dumps(cleaned_map) +','+ '\n')
                            # result.append(cleaned_map)
        # with open(self.save_clean, 'w') as f:
        #     json.dump(result, f, indent=4)

        mean_sisdr = np.mean(sisdr)
        mean_sdri = np.mean(sdr)

        return mean_sisdr, mean_sdri


if __name__ == '__main__':
    device = "cuda"
    
    configs = parse_yaml('config/audiosep_base.yaml')

    # WavCaps Evaluator
    # print(f'Evaluation on AudioSet with [caption] queries.')
    # wavcaps_evaluator = WavCapsEvaluator(datatype="AS")
    
    # print(f'Evaluation on BBC with [caption] queries.')
    # wavcaps_evaluator = WavCapsEvaluator(datatype="BBC")

    # print(f'Evaluation on Freesound with [caption] queries.')
    # wavcaps_evaluator = WavCapsEvaluator(datatype="FSD")

    print(f'Evaluation on soundbile with [caption] queries.')
    wavcaps_evaluator = WavCapsEvaluator(datatype="SB")


    # Load model
    query_encoder = CLAP_Encoder().eval()

    checkpoint_path='checkpoint/audiosep_base_4M_steps.ckpt'

    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
        query_encoder=query_encoder
    ).to(device)

    # evaluation on wavcaps
    SISDR, SDR = wavcaps_evaluator(pl_model)
    msg_wavcaps = "wavcaps Avg SDR: {:.3f}, SISDR: {:.3f}".format(SDR, SISDR)
    print(msg_wavcaps)