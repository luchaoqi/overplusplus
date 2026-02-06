import os
import sys
import glob
import json
import argparse
from tqdm import tqdm
from loguru import logger
from collections import Counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_rootdir', type=str, required=True, help='input root directory')
    parser.add_argument('--output_json', type=str, default='casper.json', help='saved json file')
    # parser.add_argument('--repeat_omnimatte_tripod', type=int, default=256, help='repeat times for omnimatte+tripod')
    parser.add_argument('--repeat_omnimatte_tripod', type=int, default=0, help='repeat times for omnimatte+tripod') # 46
    parser.add_argument('--repeat_kubric', type=int, default=0, help='repeat times for kubric') # 569
    parser.add_argument('--repeat_vos2019', type=int, default=0, help='repeat times for vos2019') # 3471
    parser.add_argument('--repeat_object_paste', type=int, default=0, help='repeat times for object paste') 
    parser.add_argument('--repeat_water', type=int, default=0, help='repeat times for water examples') # 10
    parser.add_argument('--repeat_hyoutube', type=int, default=0, help='repeat times for hyoutube examples') # 2314 subset of vos2019
    # parser.add_argument('--repeat_iharmony4', type=int, default=0, help='repeat times for iharmony4 examples')
    parser.add_argument('--repeat_omnimatte_tripod_augment', type=int, default=0, help='repeat times for omnimatte+tripod augment')
    parser.add_argument('--repeat_movie', type=int, default=0, help='repeat times for movie')
    parser.add_argument('--repeat_omnimatte_zero', type=int, default=0, help='repeat times for omnimatte zero')
    args = parser.parse_args()

    root_dir = args.input_rootdir
    output_json_path = args.output_json
    export_list = []
    count = 0
    data_list = []

    water_examples = [
        'flamingo', 'kite-surf', 'pexels-bike-puddle', 'pexels-bike-puddle2', 'pexels-boat-bridge', 
        'pexels-boat-ocean', 'pexels-skate-puddle', 'pexels-speedboat', 'pexels-suv-rain', 'pexels-towing-boat',
    ]
    water_weight = args.repeat_water  # we double the training weight of water examples for public models

    r_omnimatte_tripod = args.repeat_omnimatte_tripod
    r_kubric = args.repeat_kubric
    r_vos2019 = args.repeat_vos2019
    r_op = args.repeat_object_paste
    r_hyoutube = args.repeat_hyoutube
    # r_iharmony4 = args.repeat_iharmony4
    r_omnimatte_tripod_augment = args.repeat_omnimatte_tripod_augment
    r_movie = args.repeat_movie
    r_omnimatte_zero = args.repeat_omnimatte_zero
    # Please adjust the repeat times for each category (r_omnimatte+tripod, r_kubric, r_op) according to your needs.
    # Supppose you generate N_kubric Kubric videos, N_op ObjectPaste videos and with our released 46 omnimatte and tripod videos,
    # * N_kubric = 569 from the original released dataset
    # Total number of instances =   46 * r_omnimatte+tripod + N_kubric * r_kubric + N_op * r_op
    # the weight of each category is:
    #   Omnimatte+tripod:           46 * r_omnimatte+tripod / Total number of instances
    #   Kubric:                     N_kubric * r_kubric / Total number of instances
    #   Object Paste:               N_op * r_op / Total number of instances
    # In our paper, we make the training set contain 50% omnimatte+tripod, 48% kubric, and 2% object-paste.
    

    train_data_list = [
        ('omnimatte+tripod', r_omnimatte_tripod), 
        ('kubric', r_kubric),
        ('vos2019', r_vos2019), 
        ('object-paste', r_op), 
        # ('iharmony4', r_iharmony4),
        ('omnimatte+tripod_augment', r_omnimatte_tripod_augment),
        ('movie', r_movie),
        ('omnimatte_zero', r_omnimatte_zero),
    ]
    if args.repeat_hyoutube > 0:
        # we use hyoutube-object_0, hyoutube-object_1, hyoutube-object_2
        for object_index in ['object_0', 'object_1', 'object_2', 'object_3', 'object_4', 'object_5']:
            train_data_list.append((f'hyoutube-{object_index}', r_hyoutube))

    logger.debug(f"train_data_list: {train_data_list}")
    dataset_counter = Counter()
    for category, repeat_times in train_data_list:
        if repeat_times == 0:
            continue
        if os.path.isdir(os.path.join(root_dir, category)):
            # logger.info(f"Processing {category}...")
            caption_path = os.path.join(root_dir, category).rstrip('/') + '.json'
            assert os.path.exists(caption_path), f"Caption file {caption_path} does not exist."

            captions = json.load(open(caption_path))

            data_dirs = sorted(list(glob.glob(os.path.join(root_dir, category, '*'))))

            data_dirs = [data_dir for data_dir in data_dirs if os.path.isdir(data_dir)]
            n = 0
            for data_dir in data_dirs:
                vid = os.path.basename(data_dir)
                found = False
                for caption in captions:
                    if vid == caption['path']:
                        found = True
                        n += 1
                        break
                if found:
                    if vid in water_examples:
                        r = repeat_times * water_weight
                        print(f'water examples! {vid} {r}')
                    else:
                        r = repeat_times
                    for _ in range(r):
                        # TODO: use prompt or short_prompt
                        # data_list.append((data_dir, caption['short_prompt'].lstrip('"').rstrip('"')))
                        data_list.append((data_dir, caption['prompt'].lstrip('"').rstrip('"')))
                else:
                    print(f"Caption not found for {vid}")
            logger.info(f"Processed {category}, found {n} videos, repeated {repeat_times} times")
            dataset_counter[category] += n * repeat_times
        else:
            logger.info(f"Directory {os.path.join(root_dir, category)} does not exist.")

    total_samples = sum(dataset_counter.values())
    print("=== Dataset Composition Summary ===")
    for category, count_ in dataset_counter.items():
        percent = 100.0 * count_ / total_samples if total_samples > 0 else 0
        logger.info(f"{category:20s}: {count_:6d} ({percent:6.2f}%)")
    logger.info(f"Total samples: {total_samples}")


    for data_dir, prompt in tqdm(data_list):
        export_list.append(
            {
                "file_path": os.path.abspath(data_dir),
                "text": prompt,
                "type": "video_mask_tuple",
            }
        )
        count += 1
    print(count)
    json.dump(export_list, open(output_json_path, 'w'), indent=4)
