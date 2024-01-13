# 2023-AICOSS

[2023 AICOSS hackathon competition]((https://dacon.io/competitions/official/236201/overview/description)) with **multi-GPU training**

For single-GPU training and more information about the competition, you can use [this repository](https://github.com/seok-AI/2023-AICOSS).

## Training

### Virtual Environment Settings

You should install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) for the following codes.

```bash
git clone https://github.com/seok-AI/2023-AICOSS
cd 2023-AICOSS/
conda env create -f ddp.yaml
```

```bash
conda activate ddp
```

You can change the name of your environment by using the `-n` option such as `conda env create -f ddp.yaml -n 'your_env_name'`.

### Training example

```bash
python main.py \
	--epochs=10 \
	--lr=1e-4 \
	--min_lr=1e-7 \
	--model_name=cvt384_q2l \
	--loss_name=PartialSelectiveLoss \
	--img_size=384 \
	--batch_size=8 \
	--grad_accumulation=8 \
	--path=/data
```

You should change `--path` to your data path.

If you have less/more GPUs than 4, you can change `--gpu=0,1,2,3` to whatever you want.

And `batch_size` means for each GPU device, so if you set `--batch_size=8 --gpu=0,1`  your total batch size will be 16.

<br/>

We used 4 x Nvidia RTX 3090 GPUs for training.

## Reference

- [ML_Decoder](https://github.com/Alibaba-MIIL/ML_Decoder)
- [TResNet](https://github.com/Alibaba-MIIL/TResNet)
- [timm](https://github.com/huggingface/pytorch-image-models/tree/main/timm)
- [query2label](https://github.com/curt-tigges/query2label)
- [Pytorch Lightning](https://lightning.ai/)

### Papers

- [RandAugment: Practical automated data augmentation with a reduced search space [NeurIPS 2020]](https://arxiv.org/pdf/1909.13719.pdf)
- [TResNet: High Performance GPU-Dedicated Architecture [WACV 2021]](https://arxiv.org/pdf/2110.10955.pdf)
- [Multi-label Classification with Partial Annotations using Class-aware Selective Loss [CVPR 2022]](https://arxiv.org/pdf/2110.10955.pdf)
- [ML-Decoder: Scalable and Versatile Classification Head [WACV 2023]](https://arxiv.org/pdf/2111.12933.pdf)
