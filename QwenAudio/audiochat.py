from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import json
from tqdm import tqdm
from pedalboard.io import AudioFile
from accelerate.utils import gather_object
import torchaudio
MODEL_ID = "Qwen/Qwen-Audio-Chat"
accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)
tokenizer.padding_side = 'left'
tokenizer.pad_token_id = tokenizer.eod_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"": accelerator.process_index},
    trust_remote_code=True,
    bf16=True
).eval()

# model.generation_config.length_penalty=None
# model.generation_config.do_sample=False
# print(model.generation_config.length_penalty)
def batch_process(audios):
    
    queries = [
        "<audio>{}</audio>Describe the genre, instruments, tempo and mood of music piece in form of sentences. Do not summary answer.".format(i) for i in audios
    ]
    # queries = [
    #     "<audio>{}</audio>Describe the audio in one sentence in English.".format(i) for i in audios
    # ]
    audio_info = [tokenizer.process_audio(audio) for audio in queries]
    input_tokens = tokenizer(queries, return_tensors='pt', padding='longest',audio_info=audio_info)
    input_ids = input_tokens.input_ids
    input_len = input_ids.shape[-1]
    attention_mask = input_tokens.attention_mask
    accelerator.wait_for_everyone()
    start = time.time()
    accelerator.wait_for_everyone()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask = attention_mask.cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=77,
            num_return_sequences=1,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
            audio_info=audio_info
        )
    answers = [
        tokenizer.decode(o[input_len:].cpu(), skip_special_tokens=True, audio_info=audio_info).strip() for o in outputs
    ]

    answers_gathered = gather_object(answers)
    # end = time.time()
    # print("took: ", end - start)

    return answers_gathered



class AudioDataset(Dataset):

    def __init__(self,):
        super(AudioDataset, self).__init__()
        json_file_path = 'MTG-Jamendo/mtg_tag_filtered.json'
        with open(json_file_path, 'r') as f:
            json_obj = json.load(f)
        self.wav_paths = [item["location"] for item in json_obj]

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index):
        audio = self.wav_paths[index]
        try:
            with AudioFile(audio) as f:
                wav = f.read(f.frames)

            # wav, in_sr = torchaudio.load(audio)
        except:
            print("Error loading audio: ", audio)
            return self.__getitem__(index+1)
        return audio

def collate_fn(batch):
    audio_list = []
    for queryid in batch:
        audio_list.append(queryid)
    return audio_list

data_set = AudioDataset()

dataloder=DataLoader(data_set,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
        )


# if isinstance(model, DistributedDataParallel):
#     model = model.module
for batch in tqdm(dataloder):
    with accelerator.split_between_processes(batch) as batch:
        audios = batch
        answers = batch_process(audios)
        audios = gather_object(audios)
        if accelerator.is_main_process:
            with open('MTG.txt', 'a') as f:
                for i, a in enumerate(answers):
                    print(audios[i], a)
                    f.write(json.dumps({"audio": audios[i], "description": a}) + "\n")

# audios = [
#     '/mmu-audio-ssd/frontend/audioSep/wanghualei/code/Magnatune/f/the_headroom_project-jetuton_andawai-01-linda_morena-59-88.mp3',
#     '/mmu-audio-ssd/frontend/audioSep/wanghualei/code/Magnatune/f/the_headroom_project-jetuton_andawai-01-linda_morena-88-117.mp3',
#     '/mmu-audio-ssd/frontend/audioSep/wanghualei/code/Magnatune/9/american_bach_soloists-heinrich_schutz__musicalische_exequien-01-musicalische_exequien_swv_279_teil_i_concert_in_form_einer_teutschen_begrabnismissa-784-813.mp3',
# ]
# batch_process(audios)
# ps -ef | grep audiochat.py | grep -v grep | awk '{print $2}' | xargs kill -9
## accelerate launch --multi_gpu audiochat.py