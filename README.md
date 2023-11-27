# LLama2 Inference Project

This repository contains the implementation of Meta's LLAMA2 language model for inferencing. The implementation is built from scratch using only pytorch and standard python libs.

Model key features:
- Rotary Positional Embedding
- RMS Normalization
- KV cache
- Grouped Query Attention (GQA)
- SwiGLU activation

Also developing a range of inference methods including Greedy, Beam Search, Temperature Scaling, Random Sampling, Top-K and Top-P, providing flexibility for users to choose the optimal approach for their specific natural language processing (NLP) tasks. Currently, the Top-P inference method has been successfully integrated into the codebase.

## Installation

```
pip install -r requirements.txt
```

## Model Weights

The model weights can be downloaded from Meta's LLAMA2 model repository and then downloaded with the download.sh file. Please refer to the official repository for the most up-to-date information on weight downloads. 

## How to Run Inference

1. Clone the repository:
```
git clone https://github.com/gkpodder/Llama2-implementation.git
```

2. Download the model weights:
```
./download.sh
```

3. Run inference:
```
python inference.py
```

## Benchmarking

KV caching and GQA resulted in a significant speed up in inference of about 50% in my benchmarking. 

## Contact

For questions or concerns, please open a GitHub issue
