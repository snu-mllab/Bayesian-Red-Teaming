# Query-Efficient-Black-Box-Red-Teaming-via-Bayesian-Optimization
About Official PyTorch implementation of "Query-Efficient Black-Box Red Teaming via Bayesian Optimization" (ACL'23 Long Paper - Main Conference)

## Installation
Requirements : Anaconda, cudatoolkit 11.3
1. Create Conda Environment
```bash
    conda create -n BRT python=3.9.13 -y
    conda activate BRT
```
2. Install PyTorch
```bash
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
3. Install ParlAI
```bash
    git clone https://github.com/facebookresearch/ParlAI.git ~/ParlAI
    cd ~/ParlAI
    pip install -r requirements.txt
    python setup.py develop
```
4. Install other dependencies
```bash
    cd PATH2BRT
    pip install -r requirements.txt
```

## Run BRT
### Arguments

* --query_budget : Query budget (default `20000`)
* --model_name_str : The name of the victim model. (default `bb1-3B`)
* --user_input_pool_str : The name of the user input pool. (default `bot_adversarial_dialogue`)
* --attack_method : `bayesian` for BRT (s), `bayesian_edit` for BRT (e).
* --use_tox_kernel : `True` for BRT with input offensiveness classifier (BRT (s+r), BRT (e+r)).
* --tox_range : `whole` for generic experiments. `safe` for hard positive red teaming. (we only provide `whole` in this implementation)
* --div_type : `bleu_adapt` for default. It modifies lambda adaptive to diversity
* --div_coeff : `0.3` for BRT (s), `0.03` for BRT (e).
* --target_BLEU : the diversity budget D.

### Experiments on Bot Adversarial Dialogue user input pool.
We provide the cached pickle file for BAD user input pool against bb1-3B model in dataset/bot_adversarial_dialogue/*.pkl.

Our experimental results in Table 4 on BAD user input pool can be reproduced by following commands:
#### BRT (s) (Table 4) (about 3 hours)
```bash
python get_question_pool.py --attack_method bayesian --use_tox_kernel False --div_coeff 0.3 --target_BLEU 42.0
```
#### BRT (s+r) (Table 4) (about 3 hours)
```bash
python get_question_pool.py --attack_method bayesian --use_tox_kernel True --div_coeff 0.3 --target_BLEU 40.5
```
#### BRT (e) (Table 4) (about 15 hours)
```bash
python get_question_pool.py --attack_method bayesian_edit --use_tox_kernel False --div_coeff 0.03 --target_BLEU 40.5
```
#### BRT (e+r) (Table 4) (about 15 hours)
```bash
python get_question_pool.py --attack_method bayesian_edit --use_tox_kernel True --div_coeff 0.03 --target_BLEU 40.5
```

## Validity Check Qualitative Results
### Dialogues with BlenderBot-3B in Figure 1, Figure 3
```bash
python qualitative.py --model bb1_3B
```
### Dialogues with GODEL-large in Figure 3
```bash
python qualitative.py --model GODEL-large
```


## Machine Information
Below are the information about machine that authors used.
* OS: Ubuntu 16.04
* CUDA Driver Version: 465.19.01
* gcc: 5.4.0
* nvcc(CUDA): 11.3
* CPU: AMD EPYC 7402 24-Core Processor
* GPU: NVIDIA GeForce RTX 3090 GPU
