#  💡 Music-TAG-Generation
We use BEATs model to acquire tags and utilize LLM to expand into captions
## 🚀 model

1. Download pretrain BEATs weight from [BEATs](https://github.com/microsoft/unilm/tree/master/beats)

2. BEATs model to classfier
```
accelerate config
accelerate launch --multi_gpu python classfier.py
```

3. LLM(such as GPT4 or kimi) to expand into captions
```
python tag_caption.py
```

## 🚢 Datasets

Expand the mtg dataset. You can download [mtg-jamendo-dataset](https://mtg.github.io/mtg-jamendo-dataset/) and get raw_30s folder.


### License
This project is licensed under the MIT License.
