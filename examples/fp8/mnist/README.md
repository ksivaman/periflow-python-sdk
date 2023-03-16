# Basic MNIST Example with optional FP8 using TransformerEngine

[TransformerEngine](https://github.com/NVIDIA/TransformerEngine) is a library for accelerating Transformer models on NVIDIA GPUs, including using 8-bit floating point (FP8) precision on Hopper GPUs.


```bash
python main.py
python main.py --use-te   # Linear layers from TransformerEngine
python main.py --use-fp8  # FP8 + TransformerEngine for Linear layers
```

