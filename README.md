# Code for MACRec (AAAI 2026)

This repo is for source code of AAAI 2026 paper "Multi-Aspect Cross-modal Quantization for Generative Recommendation".


### Pseudo-label Generation

```
cd data
python balance_kmeans.py
```

### Training the Cross-modal Quantitative RQ-VAE
```
cd cross_index
bash script/run_rqvae.sh          # Run training  
bash script/gen.sh # Generate code  
```


### Fine-tuning
```
bash script/run_finetune.sh
```


### Citation
```
@article{zhang2025multi,
  title={Multi-Aspect Cross-modal Quantization for Generative Recommendation},
  author={Zhang, Fuwei and Liu, Xiaoyu and Xi, Dongbo and Yin, Jishen and Chen, Huan and Yan, Peng and Zhuang, Fuzhen and Zhang, Zhao},
  journal={arXiv preprint arXiv:2511.15122},
  year={2025}
}
```

### Acknowledgment
This code is based on https://github.com/zhaijianyang/MQL4GRec. 