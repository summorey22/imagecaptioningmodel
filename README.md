# Scene Description using GPT-2   
---
## Source code and documentation for the model which generates captions for any given image. 
---
## Dataset:
MSCOCO dataset: n_samples = 13500  

For geting dataset:
```bash
bash getData.sh
```
---
## Environment:
We recommend anaconda/miniconda.   
You can setup your environment(x86_64) by entering:

```bash
conda env create -f environment.yml
```

If you want to use pip, enter following in your venv:
```bash
pip install -r requirements.txt
```

---
## Training:
```bash
python train.py --p /path/to/training/data --a /path/to/annontations 
```
If you are on Ubuntu or similar distribution juse `python` with `python3`


It also supports some more flags which you can see by entering   
```bash
python train.py --help
```
---
## Execution
```bash
python scene_captioning.py 
```
