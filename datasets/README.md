# Training Data for an Over++ model

> We are going through internal paperwork process to fully respect the copyrights before releasing the dataset. You may use the instructions below to reproduce the dataset.

### Table of Contents:
- [Overview](#overview)
- [Omnimatte & Tripod](#omnimatte&tripod)
- [Kubric](#kubric)
- [Movies](#movies)
- [OmnimatteZero](#omnimattezero)
- [Masks](#masks)
- [Captions](#captions)
- [Omnimatte & Tripod Augmented](#omnimatte&tripod_augmented)

## Overview <a name="overview"></a>
We follow the same structure as [gen-omnimatte](https://github.com/gen-omnimatte/gen-omnimatte-public/tree/main/datasets). The `datasets` folder contains the following structure:

```
./train-casper
├── casper.json                                 # final captions for the whole dataset
├── kubric
│   ├── kubric-0000000                          # example data
│   │   ├── mask.mp4                            # video mask
│   │   ├── rgb_full.mp4                        # video without effects
│   │   └── rgb_removed.mp4                     # video with effects
├── kubric.json
├── movie
├── movie.json
├── omnimatte+tripod
│   ├── bmx-bumps                               # example data
│   │   ├── bg                                  # images without effects
│   │   ├── input                               # images with effects
│   │   └── trimask                             # images for effect region
├── omnimatte+tripod.json
├── omnimatte+tripod_augment
├── omnimatte+tripod_augment.json
├── omnimatte_zero
└── omnimatte_zero.json
```

After downloading all the data below, please run the preprocessing script in `helper_gen_omnimatte_data.ipynb` (Search for "handle data for our model training (1st round preprocessing)" in the file [here](helper_gen_omnimatte_data.ipynb)).


After finishing generating captions for each dataset, you may merge them into a single json file:
```bash
python datasets/merge_training_data.py --input_rootdir ./train-casper --repeat_omnimatte_tripod 512 --repeat_kubric 40 --repeat_water 32 --repeat_omnimatte_tripod_augment 64 --repeat_movie 2048 --repeat_omnimatte_zero 1024 --output_json "./train-casper/casper.json"
```


## Omnimatte & Tripod <a name="omnimatte&tripod"></a>
- Please download the data from the google drive [here](https://drive.google.com/drive/folders/1qnIduW8v9U6cobOyYyDP3yGcxo1PjU97?usp=sharing).

We made additional changes to the original datasets:

- Fix resolution error for certain videos
- Manually adjust training captions after MLLM inference to draw more attention to water examples (splashing) - this is motivated by the paper [force-prompting](https://force-prompting.github.io/) ("Wind" keyword is important at train time but not at inference time)

> You may also download the original datasets from DAVI-2017 and run [gen-omnimatte](https://github.com/gen-omnimatte/gen-omnimatte-public) inference on them to generate the training data.

## Kubric <a name="kubric"></a>
- Please download the data from the google drive [here](https://drive.google.com/drive/folders/1qnIduW8v9U6cobOyYyDP3yGcxo1PjU97?usp=sharing).

Details of generating kubric data can be found [here](https://github.com/gen-omnimatte/gen-omnimatte-public/tree/main/datasets)


## Movies <a name="movies"></a>
- Please download the data from [OmnimatteRF: Robust Omnimatte with 3D Background Modeling](https://omnimatte-rf.github.io/)


## OmnimatteZero <a name="omnimattezero"></a>
- Please reach out to authors of [OmnimatteZero: Training-free Real-time Omnimatte with Pre-trained Video Diffusion Models](https://dvirsamuel.github.io/omnimattezero.github.io/), we also provide samples of the dataset [here](https://drive.google.com/file/d/1g471d0vZh6My4Zu68qvVVsm2QAxp0nIb/view?usp=sharing).

## Masks <a name="masks"></a>
Please run the cmd below to generate masks:
```bash
python datasets/helper_gen_omnimatte_mask.py --mode <dataset_name>
```
Note that you may need to manually clean / adjust the threshold for the mask generation depending on the dataset.

## Captions <a name="captions"></a>
**Update**: Please refer to SOTA VLM (e.g. [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)) for better caption generation.

You may run the cmd below to generate captions:
```bash
python datasets/caption_videos.py --input_rootdir <path to dataset> --output_json <path to output json>
```

You may also run the cmd below for larger dataset:
```bash
python datasets/caption_videos_helper.py --instance_data_root <path to dataset> --gpus 0,1,2,3
```
> please adjust the arguments in the script to fit your environment

## Omnimatte & Tripod Augmented <a name="omnimatte&tripod_augmented"></a>
Please run the cmd below to generate augmented data:
```bash
# augment captions
python datasets/caption_augmentation.py
# generate videos
python helper_gen_omnimatte_inference_multi_gpu.py --output_dir '/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/train-casper/omnimatte+tripod_augment' --n_gpus 8 --mode t2v
```
> please adjust the arguments in the script to fit your environment