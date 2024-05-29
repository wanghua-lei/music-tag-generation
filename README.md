# Music-TAG-Generation
We use BEATs model to acquire tags and utilize LLM to expand into captions
## 🎵 Model

1. Download pretrain BEATs weight from [BEATs](https://github.com/microsoft/unilm/tree/master/beats)

2. BEATs model to classfier
```
accelerate config
accelerate launch --multi_gpu python classfier.py
python generate_tag.py
```

3. LLM(such as GPT4 or deepseek) to expand into captions
```
python gpt/tag_caption.py
find /path -type f > output.txt
```


## 🔥 Datasets

Download the mtg dataset. You can download [mtg-jamendo-dataset](https://mtg.github.io/mtg-jamendo-dataset/) and get raw_30s 55,701 tracks.
https://huggingface.co/datasets/wanghappy/Music-tag-generation/

### License
This project is licensed under the MIT License.
