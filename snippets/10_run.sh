pip install torch

# 1. check the maths (about ten seconds)
python mamba_lm.py test

# 2. train: downloads Tiny Shakespeare, saves mamba.pt
python mamba_lm.py train

#    the same with the Mamba-2 block, or on your own UTF-8 text
python mamba_lm.py train --arch mamba2
python mamba_lm.py train --data my_corpus.txt

# 3. generate from the checkpoint
python mamba_lm.py generate --prompt "ROMEO:" --max-new 300
