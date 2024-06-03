
# robust-model-watermarking
This is the official implementation of our paper Towards Robust Model Watermark via Reducing Parametric Vulnerability, accepted by the International Conference on Computer Vision (ICCV), 2023. We will release our codes soon.

## Train vanilla watermarked models
To watermark the model using "Content" watermark samples, run the following code.
```CUDA_VISIBLE_DEVICES=0 python train.py --seeds=[0,1,2] --method=STD --model=ResNet18 -wt=test```

Checkpoints and and training logs can be found in `dfs/cifar10_y0_test_40000_400_c1.00e+00/STD_wd5.00e-04/ResNet18`

To use "Noise" or "Unrelated" watermark samples, replace ``-wt=test`` with ``-wt=gauss -t=0.1`` or ``-wt=svhn`` respectively

## Train APP watermarked model
```CUDA_VISIBLE_DEVICES=0 python train.py --seeds=[0,1,2] --method=APP --alpha=1e-2 --app-eps=2e-2 --model=ResNetCBN18 -wt=test```

Checkpoints and and training logs can be found in 
`dfs/cifar10_y0_test_40000_400_c1.00e+00/APP_a1.00e-02_rl2_eps2.00e-02_pbs64_bbs64_wd5.00e-04/ResNetCBN18`

## Evaluate the robustness of the watermarked models

```
CUDA_VISIBLE_DEVICES=0 python attack.py --seeds=[0,1,2] --method=FT --name=FT_E30_LR5E-2_DROP --ft-lr=5e-2 --ft-lr-gamma=0.5 --ft-lr-drop=[5,10,15,20,25] --ft-max-epoch=30 --target-dir=RESULT_DIR

CUDA_VISIBLE_DEVICES=0 python attack.py --seeds=[0,1,2] --method=FP --name=FP_FINE_E30_LR5E-2_DROP --ft-lr=5e-2 --ft-lr-gamma=0.5 --ft-lr-drop=[5,10,15,20,25] --prune-rate='np.arange(0.8,1,0.05)' --ft-max-epoch=30 --target-dir=RESULT_DIR

CUDA_VISIBLE_DEVICES=0 python attack.py --seeds=[0,1,2] --method=ANP --name=ANP_E30 --anp-max-epoch=30 --target-dir=RESULT_DIR

CUDA_VISIBLE_DEVICES=0 python attack.py --seeds=[0,1,2] --method=NAD --name=NAD_TS2E-2_E10 --ft-max-epoch=10 --ft-lr=2e-2 --ft-batch-size=64 --ft-lr-drop=[2,4,6,8] --nad-max-epoch=10 --nad-lr=2e-2 --nad-batch-size=64 --nad-lr-drop=[2,4,6,8] --target-dir=RESULT_DIR

CUDA_VISIBLE_DEVICES=0 python attack.py --seeds=[0,1,2] --method=MCR --name=MCR_E100_LR1E-2_E50_LR5E-2DROP --ft-max-epoch=50 --mcr-lr=1e-2 --mcr-max-epoch=100 --ft-lr=5e-2 --ft-lr-drop=[10,20,30,40] --target-dir=RESULT_DIR

CUDA_VISIBLE_DEVICES=0 python attack.py --seeds=[0,1,2] --method=NNL --name=NNL --nc-epoch=15 --ft-max-epoch=15 --ft-lr=2e-2 --ft-lr-drop=[10,] --target-dir=RESULT_DIR
```
Replace the "RESULT_DIR" with the directory of the checkpoints, like `dfs/cifar10_y0_test_40000_400_c1.00e+00/STD_wd5.00e-04/ResNet18`

The attack results can be found in "RESULT_DIR"
