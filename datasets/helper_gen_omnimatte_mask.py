import os
import mediapy as media
import numpy as np
from skimage.color import rgb2lab, deltaE_cie76, deltaE_ciede2000
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_closing, disk, binary_dilation, binary_erosion, binary_opening, remove_small_objects
from scipy.ndimage import median_filter
import shutil
from skimage.morphology import remove_small_holes  # Added for hole filling
from skimage.measure import label  # Added for connected components to remove small noise
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os, cv2
# need to run script in helper_gen_omnimatte_data.ipynb first before running this script
# TODO: use cv2.erode and cv2.dilate instead of skimage.morphology.binary_opening and skimage.morphology.binary_closing -> check helper_paper_figure.ipynb

def stabilize_mask_sequence(mask_frames,
                            temporal_window: int = 5,
                            min_area: int = 200,
                            open_radius: int = 1,
                            close_radius: int = 1,
                            to_uint8: bool = False):
    """
    Args:
        mask_frames: Sequence/list/array of shape [T, H, W], bool or {0,1}/{0,255}
        temporal_window: Odd window for temporal median filter (majority vote over time), larger window for more stable mask
        min_area: Remove connected components smaller than this per frame
        open_radius: Spatial opening disk radius (remove speckles)
        close_radius: Spatial closing disk radius (fill small holes)
        to_uint8: Return {0,255} uint8 if True, else boolean
    Returns:
        np.ndarray of shape [T, H, W]
    """
    assert temporal_window % 2 == 1, "temporal_window should be odd for median filtering"

    seq = np.asarray(mask_frames)
    # Normalize to boolean
    if seq.dtype != bool:
        seq = (seq > (seq.max() / 2.0 if seq.max() > 1 else 0.5))

    # Temporal median filter to stabilize flicker
    seq_med = median_filter(seq.astype(np.uint8), size=(temporal_window, 1, 1)) >= 1

    # Spatial cleanup per frame
    T, H, W = seq_med.shape
    se_open = disk(open_radius) if open_radius > 0 else None
    se_close = disk(close_radius) if close_radius > 0 else None

    out = np.empty_like(seq_med)
    for t in range(T):
        m = seq_med[t]
        if se_open is not None:
            m = binary_opening(m, footprint=se_open)
        if se_close is not None:
            m = binary_closing(m, footprint=se_close)
        if min_area > 0:
            m = remove_small_objects(m, min_size=min_area)
        out[t] = m

    if to_uint8:
        return (out.astype(np.uint8) * 255)
    return out

def threshold_mask(mask, sigma=1, threshold=None, otsu_weight=1, dilate_radius=3):
    # apply sequence of operations to the mask
    mask = gaussian(mask, sigma=sigma)
    threshold = threshold_otsu(mask) if threshold is None else threshold
    mask = mask >= threshold * otsu_weight
    mask = binary_opening(mask, footprint=disk(dilate_radius))
    mask = binary_closing(mask, footprint=disk(dilate_radius))
    mask = (mask * 255).astype(np.uint8)
    return mask
    
def iterate_over_img_dir(dir_path):
    IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif', '.ico', '.heic', '.heif']
    # upper case
    IMG_EXTENSIONS_UPPER = [ext.upper() for ext in IMG_EXTENSIONS]
    IMG_EXTENSIONS = IMG_EXTENSIONS + IMG_EXTENSIONS_UPPER
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if any(file.endswith(ext) for ext in IMG_EXTENSIONS):
                yield os.path.join(root, file)

def create_kubric_mask(in_dir, out_dir, file_name):
    os.makedirs(out_dir, exist_ok=True)
    file = file_name
    otsu_weight = 1
    dilate_radius = 3
    N_largest_components = 10
    min_size_threshold = 20

    if os.path.isdir(os.path.join(in_dir, file)):
        mask_path = os.path.join(in_dir, file, 'mask.mp4')
        rgb_full_path = os.path.join(in_dir, file, 'rgb_full.mp4')
        rgb_removed_path = os.path.join(in_dir, file, 'rgb_removed.mp4')

        mask = media.read_video(mask_path)
        fps = mask.metadata.fps
        rgb_full = media.read_video(rgb_full_path)
        rgb_removed = media.read_video(rgb_removed_path)


        # ----------------------------------- mask ----------------------------------- #
        # mask = media.read_video(mask_path)
        # mask = np.where(np.logical_and(mask >63, mask < 192), 127, mask)
        # mask = np.where(mask >= 192, 255, mask)
        # mask = np.where(mask <= 63, 0, mask)
        # # ! remap color (fg: black, bg: white)
        # mask = np.where(mask == 255, 127, mask) # remove other foreground, map them to gray
        # mask = np.where(mask == 127, 255, mask) # background to white
        mask = np.zeros_like(mask)

        # ------------------------------------ rgb ----------------------------------- #
        rgb_full_lab = rgb2lab(rgb_full)
        rgb_removed_lab = rgb2lab(rgb_removed)

        delta_e = deltaE_cie76(rgb_full_lab, rgb_removed_lab)
        delta_e_base = deltaE_cie76(rgb_full_lab, np.zeros_like(rgb_full_lab))
        delta_e_relative = delta_e / (delta_e_base + 1e-6)

        delta_e_threshold = threshold_otsu(delta_e)
        delta_e_relative_threshold = threshold_otsu(delta_e_relative)

        # Process per frame to create binary mask
        mask_frames = []
        for i in range(delta_e.shape[0]):
            absolute_mask = threshold_mask(delta_e[i], dilate_radius=dilate_radius, otsu_weight=otsu_weight, threshold=delta_e_threshold)
            relative_mask = threshold_mask(delta_e_relative[i], dilate_radius=dilate_radius, otsu_weight=otsu_weight, threshold=delta_e_relative_threshold)
            merge_mask = np.logical_or(absolute_mask, relative_mask)
            merge_mask = threshold_mask(merge_mask, dilate_radius=dilate_radius, otsu_weight=otsu_weight)

            # keep N largest components
            labeled_mask, num_labels = label(merge_mask, return_num=True)
            if num_labels > 0:
                component_sizes = np.bincount(labeled_mask.ravel())[1:] # ignore background (label 0)
                original_labels = np.arange(1, num_labels + 1)
                large_enough_indices = np.where(component_sizes >= min_size_threshold)[0]
                if len(large_enough_indices) > 0:
                    candidate_labels = original_labels[large_enough_indices]
                    candidate_sizes = component_sizes[large_enough_indices]
                    num_to_keep = min(N_largest_components, len(candidate_labels))
                    largest_indices = np.argsort(candidate_sizes)[-num_to_keep:]
                    final_labels = candidate_labels[largest_indices]
                    merge_mask = np.isin(labeled_mask, final_labels)
                else:
                    merge_mask = np.zeros_like(merge_mask, dtype=bool)

            merge_mask = binary_dilation(merge_mask, footprint=disk(dilate_radius))
            
            # ! remap color
            # merge_mask = np.where(merge_mask, 127, 0)
            # merge_mask = np.stack([merge_mask, merge_mask, merge_mask], axis=-1).astype(np.uint8)
            # final_mask = np.where(merge_mask == 127, 127, mask[i])
            final_mask = merge_mask
            mask_frames.append(final_mask)

        # copy the rgb and save mask
        save_name = os.path.basename(file)
        save_path = os.path.join(out_dir, save_name)
        os.makedirs(save_path, exist_ok=True)
        shutil.copy(rgb_full_path, os.path.join(save_path, 'rgb_full.mp4'))
        shutil.copy(rgb_removed_path, os.path.join(save_path, 'rgb_removed.mp4'))
        media.write_video(os.path.join(save_path, 'mask.mp4'), np.stack(mask_frames), fps=fps)
        print(f'Finished {save_name}')



def create_rose_mask(in_dir, out_dir, file_name):
    os.makedirs(out_dir, exist_ok=True)
    file = file_name
    otsu_weight = 1.2
    dilate_radius = 7
    N_largest_components = 10
    min_size_threshold = 10
    save_name = os.path.basename(file)
    save_path = os.path.join(out_dir, save_name)
    if os.path.exists(save_path):
        print(f'File {save_name} already exists')
        return

    if os.path.isdir(os.path.join(in_dir, file)):
        mask_path = os.path.join(in_dir, file, 'mask.mp4')
        rgb_full_path = os.path.join(in_dir, file, 'rgb_full.mp4')
        rgb_removed_path = os.path.join(in_dir, file, 'rgb_removed.mp4')

        # mask = media.read_video(mask_path)
        # fps = mask.metadata.fps
        rgb_full = media.read_video(rgb_full_path)
        rgb_removed = media.read_video(rgb_removed_path)
        fps = rgb_full.metadata.fps

        # ----------------------------------- mask ----------------------------------- #
        # mask = media.read_video(mask_path)
        # mask = np.where(np.logical_and(mask >63, mask < 192), 127, mask)
        # mask = np.where(mask >= 192, 255, mask)
        # mask = np.where(mask <= 63, 0, mask)
        # # ! remap color (fg: black, bg: white)
        # mask = np.where(mask == 255, 127, mask) # remove other foreground, map them to gray
        # mask = np.where(mask == 127, 255, mask) # background to white
        # mask = np.zeros_like(mask)

        # ------------------------------------ rgb ----------------------------------- #
        rgb_full_lab = rgb2lab(rgb_full)
        rgb_removed_lab = rgb2lab(rgb_removed)

        delta_e = deltaE_ciede2000(rgb_full_lab, rgb_removed_lab)
        # delta_e_base = deltaE_ciede2000(rgb_full_lab, np.zeros_like(rgb_full_lab))
        # delta_e_relative = delta_e / (delta_e_base + 1e-6)

        delta_e_threshold = threshold_otsu(delta_e)
        # delta_e_relative_threshold = threshold_otsu(delta_e_relative)

        # Process per frame to create binary mask
        mask_frames = []
        for i in range(delta_e.shape[0]):
            absolute_mask = threshold_mask(delta_e[i], dilate_radius=dilate_radius, otsu_weight=otsu_weight, threshold=delta_e_threshold)
            # relative_mask = threshold_mask(delta_e_relative[i], dilate_radius=dilate_radius, otsu_weight=otsu_weight, threshold=delta_e_relative_threshold)
            # merge_mask = np.logical_or(absolute_mask, relative_mask)
            # merge_mask = threshold_mask(merge_mask, dilate_radius=dilate_radius, otsu_weight=otsu_weight)
            
            merge_mask = absolute_mask
            # keep N largest components
            labeled_mask, num_labels = label(merge_mask, return_num=True)
            if num_labels > 0:
                component_sizes = np.bincount(labeled_mask.ravel())[1:] # ignore background (label 0)
                original_labels = np.arange(1, num_labels + 1)
                large_enough_indices = np.where(component_sizes >= min_size_threshold)[0]
                if len(large_enough_indices) > 0:
                    candidate_labels = original_labels[large_enough_indices]
                    candidate_sizes = component_sizes[large_enough_indices]
                    num_to_keep = min(N_largest_components, len(candidate_labels))
                    largest_indices = np.argsort(candidate_sizes)[-num_to_keep:]
                    final_labels = candidate_labels[largest_indices]
                    merge_mask = np.isin(labeled_mask, final_labels)
                else:
                    merge_mask = np.zeros_like(merge_mask, dtype=bool)

            
            # merge_mask = binary_dilation(merge_mask, footprint=disk(dilate_radius))
            
            # ! remap color
            # merge_mask = np.where(merge_mask, 127, 0)
            # merge_mask = np.stack([merge_mask, merge_mask, merge_mask], axis=-1).astype(np.uint8)
            # final_mask = np.where(merge_mask == 127, 127, mask[i])
            
            final_mask = merge_mask
            mask_frames.append(final_mask)

        mask_frames = stabilize_mask_sequence(
            mask_frames,
            temporal_window=13,
            min_area=200,
            open_radius=1,
            close_radius=1,
            to_uint8=True,  # set False to get boolean masks
        )
        # copy the rgb and save mask
        save_name = os.path.basename(file)
        save_path = os.path.join(out_dir, save_name)
        if os.path.exists(save_path):
            print(f'File {save_name} already exists')
            del rgb_full, rgb_removed, mask_frames, delta_e, delta_e_threshold, rgb_full_lab, rgb_removed_lab
            return
        
        os.makedirs(save_path, exist_ok=True)
        shutil.copy(rgb_full_path, os.path.join(save_path, 'rgb_full.mp4'))
        shutil.copy(rgb_removed_path, os.path.join(save_path, 'rgb_removed.mp4'))
        media.write_video(os.path.join(save_path, 'mask.mp4'), np.stack(mask_frames), fps=fps)
        print(f'Finished {save_name}')

        del rgb_full, rgb_removed, mask_frames, delta_e, delta_e_threshold, rgb_full_lab, rgb_removed_lab


def create_omnimatte_zero_mask(in_dir, out_dir, file_name):
    out_path = os.path.join(out_dir, file_name)
    os.makedirs(out_path, exist_ok=True)
    otsu_weight = 0.8
    dilate_radius = 20
    N_largest_components = 2
    min_size_threshold = 100

    if os.path.isdir(os.path.join(in_dir, file_name)):
        rgb_full_path = os.path.join(in_dir, file_name, 'rgb_full.mp4')
        rgb_removed_path = os.path.join(in_dir, file_name, 'rgb_removed.mp4')

        rgb_full = media.read_video(rgb_full_path)
        rgb_removed = media.read_video(rgb_removed_path)

        fps = rgb_full.metadata.fps

        # ------------------------------------ rgb ----------------------------------- #
        rgb_full_lab = rgb2lab(rgb_full)
        rgb_removed_lab = rgb2lab(rgb_removed)

        delta_e = deltaE_ciede2000(rgb_full_lab, rgb_removed_lab)
        delta_e_threshold = threshold_otsu(delta_e)

        # Process per frame to create binary mask
        mask_frames = []
        for i in range(delta_e.shape[0]):
            absolute_mask = threshold_mask(delta_e[i], dilate_radius=dilate_radius, otsu_weight=otsu_weight, threshold=delta_e_threshold)

            # keep N largest components
            labeled_mask, num_labels = label(absolute_mask, return_num=True)
            if num_labels > 0:
                component_sizes = np.bincount(labeled_mask.ravel())[1:] # ignore background (label 0)
                original_labels = np.arange(1, num_labels + 1)
                large_enough_indices = np.where(component_sizes >= min_size_threshold)[0]
                if len(large_enough_indices) > 0:
                    candidate_labels = original_labels[large_enough_indices]
                    candidate_sizes = component_sizes[large_enough_indices]
                    num_to_keep = min(N_largest_components, len(candidate_labels))
                    largest_indices = np.argsort(candidate_sizes)[-num_to_keep:]
                    final_labels = candidate_labels[largest_indices]
                    absolute_mask = np.isin(labeled_mask, final_labels)
                else:
                    absolute_mask = np.zeros_like(absolute_mask, dtype=bool)

            absolute_mask = binary_dilation(absolute_mask, footprint=disk(dilate_radius))
            # ! remap color
            # merge_mask = np.where(merge_mask, 127, 0) 
            # merge_mask = np.stack([merge_mask, merge_mask, merge_mask], axis=-1).astype(np.uint8)
            # final_mask = np.where(merge_mask == 127, 127, mask[i])
            
            final_mask = absolute_mask
            mask_frames.append(final_mask)

        mask_frames = stabilize_mask_sequence(
            mask_frames,
            temporal_window=21,
            min_area=200,
            open_radius=1,
            close_radius=1,
            to_uint8=True,  # set False to get boolean masks
        )
        media.write_video(os.path.join(out_path, 'mask.mp4'), np.stack(mask_frames), fps=fps)
        print(f'Finished {file_name}')

        del rgb_full, rgb_removed, mask_frames, delta_e, delta_e_threshold, rgb_full_lab, rgb_removed_lab

def create_movie_mask(in_dir, out_dir, file_name):
    os.makedirs(out_dir, exist_ok=True)
    dir_name = file_name
    otsu_weight = 1.2
    dilate_radius = 1
    
    if os.path.isdir(os.path.join(in_dir, dir_name)):
        parent_dir = os.path.join(in_dir, dir_name)
    else:
        return
    out_parent_dir = os.path.join(out_dir, dir_name)
    os.makedirs(os.path.join(out_parent_dir, 'input'), exist_ok=True)
    os.makedirs(os.path.join(out_parent_dir, 'bg'), exist_ok=True)
    os.makedirs(os.path.join(out_parent_dir, 'trimask'), exist_ok=True)
    
    rgb_full_dir = os.path.join(parent_dir, 'input')
    rgb_removed_dir = os.path.join(parent_dir, 'bg')
    # mask_dir = os.path.join(parent_dir, 'trimask')

    rgb_full_paths = sorted(list(iterate_over_img_dir(rgb_full_dir)))
    rgb_removed_paths = sorted(list(iterate_over_img_dir(rgb_removed_dir)))
    # mask_paths = sorted(list(iterate_over_img_dir(mask_dir)))
    _rgb_full_video = [media.read_image(rgb_full_path) for rgb_full_path in rgb_full_paths]
    _rgb_removed_video = [media.read_image(rgb_removed_path) for rgb_removed_path in rgb_removed_paths]
    # _mask_video = [media.read_image(mask_path) for mask_path in mask_paths]
    _rgb_full_video = rgb2lab(_rgb_full_video)
    _rgb_removed_video = rgb2lab(_rgb_removed_video)
    _dilate_e = deltaE_cie76(_rgb_full_video, _rgb_removed_video)
    _dilate_e_base = deltaE_cie76(_rgb_full_video, np.zeros_like(_rgb_full_video))
    _dilate_e_relative = _dilate_e / (_dilate_e_base + 1e-6)
    _dilate_e_threshold = threshold_otsu(_dilate_e)
    _dilate_e_relative_threshold = threshold_otsu(_dilate_e_relative)


    for (rgb_full_path, rgb_removed_path) in zip(rgb_full_paths, rgb_removed_paths):
        rgb_full = media.read_image(rgb_full_path)
        rgb_removed = media.read_image(rgb_removed_path)
        # mask = media.read_image(mask_path)

        # ----------------------------------- mask ----------------------------------- #
        # mask = np.where(np.logical_and(mask >63, mask < 192), 127, mask)
        # mask = np.where(mask >= 192, 255, mask)
        # mask = np.where(mask <= 63, 0, mask)
        # # ! remap color (fg: black, bg: white)
        # mask = np.where(mask == 255, 127, mask) # remove other foreground, map them to gray
        # mask = np.where(mask == 127, 255, mask) # background to white
        # mask = np.zeros_like(mask)
        # if len(mask.shape) == 2:
        #     mask = np.stack([mask, mask, mask], axis=-1)

        mask = np.zeros_like(rgb_full)
        if len(mask.shape) == 2:
            mask = np.stack([mask, mask, mask], axis=-1)
        rgb_full_lab = rgb2lab(rgb_full)
        rgb_removed_lab = rgb2lab(rgb_removed)
        delta_e = deltaE_cie76(rgb_full_lab, rgb_removed_lab)

        # Relative delta: normalize by perceptual magnitude
        delta_e_base = deltaE_cie76(rgb_full_lab, np.zeros_like(rgb_full_lab))
        delta_e_relative = delta_e / (delta_e_base + 1e-6)
        relative_mask = threshold_mask(delta_e_relative, dilate_radius=dilate_radius, otsu_weight=otsu_weight, threshold=_dilate_e_relative_threshold)

        # Absolute delta
        absolute_mask = threshold_mask(delta_e, dilate_radius=dilate_radius, otsu_weight=otsu_weight, threshold=_dilate_e_threshold)

        # merge relative and absolute delta
        merge_mask = np.logical_or(absolute_mask, relative_mask)
        merge_mask = threshold_mask(merge_mask, dilate_radius=dilate_radius, otsu_weight=otsu_weight)
        # keep the N largest components
        N_largest_components = 5
        min_size_threshold = 20
        labeled_mask, num_labels = label(merge_mask, return_num=True)
        if num_labels > 0:
            component_sizes = np.bincount(labeled_mask.ravel())[1:] # ignore background (label 0)
            original_labels = np.arange(1, num_labels + 1)
            large_enough_indices = np.where(component_sizes >= min_size_threshold)[0]
            if len(large_enough_indices) > 0:
                candidate_labels = original_labels[large_enough_indices]
                candidate_sizes = component_sizes[large_enough_indices]
                num_to_keep = min(N_largest_components, len(candidate_labels))
                largest_indices = np.argsort(candidate_sizes)[-num_to_keep:]
                final_labels = candidate_labels[largest_indices]
                merge_mask = np.isin(labeled_mask, final_labels)
            else:
                merge_mask = np.zeros_like(merge_mask, dtype=bool)
        # dilate
        merge_mask = binary_dilation(merge_mask, footprint=disk(5))

        # ! remap color (effect: gray)
        # merge_mask = np.where(merge_mask, 127, 0)
        # merge_mask = np.stack([merge_mask, merge_mask, merge_mask], axis=-1).astype(np.uint8)
        # final_mask = np.where(merge_mask == 127, 127,mask)
        final_mask = merge_mask
        
        # copy the rgb and save mask
        mask_name = os.path.basename(rgb_full_path)
        mask_path = os.path.join(out_parent_dir, 'trimask', mask_name)
        media.write_image(mask_path, final_mask)

        # shutil.copy(rgb_full_path, os.path.join(out_parent_dir, 'input', os.path.basename(rgb_full_path)))
        # shutil.copy(rgb_removed_path, os.path.join(out_parent_dir, 'bg', os.path.basename(rgb_removed_path)))
    print(f'Finished {dir_name}')


def create_omnimatte_mask(in_dir, out_dir, file_name):
    os.makedirs(out_dir, exist_ok=True)
    dir_name = file_name
    # ! data of reflection* need to be manually adjusted instead of using the default filtering parameters
    reflection_dir_names = ['reflection0', 'reflection1', 'reflection2']
    if dir_name in reflection_dir_names:
        otsu_weight = 0.4
    else:
        otsu_weight = 1.2
    dilate_radius = 1
    
    if os.path.isdir(os.path.join(in_dir, dir_name)):
        parent_dir = os.path.join(in_dir, dir_name)
    else:
        return
    out_parent_dir = os.path.join(out_dir, dir_name)
    os.makedirs(os.path.join(out_parent_dir, 'input'), exist_ok=True)
    os.makedirs(os.path.join(out_parent_dir, 'bg'), exist_ok=True)
    os.makedirs(os.path.join(out_parent_dir, 'trimask'), exist_ok=True)
    
    rgb_full_dir = os.path.join(parent_dir, 'input')
    rgb_removed_dir = os.path.join(parent_dir, 'bg')
    mask_dir = os.path.join(parent_dir, 'trimask')

    rgb_full_paths = sorted(list(iterate_over_img_dir(rgb_full_dir)))
    rgb_removed_paths = sorted(list(iterate_over_img_dir(rgb_removed_dir)))
    mask_paths = sorted(list(iterate_over_img_dir(mask_dir)))
    _rgb_full_video = [media.read_image(rgb_full_path) for rgb_full_path in rgb_full_paths]
    _rgb_removed_video = [media.read_image(rgb_removed_path) for rgb_removed_path in rgb_removed_paths]
    _mask_video = [media.read_image(mask_path) for mask_path in mask_paths]
    _rgb_full_video = rgb2lab(_rgb_full_video)
    _rgb_removed_video = rgb2lab(_rgb_removed_video)
    _dilate_e = deltaE_cie76(_rgb_full_video, _rgb_removed_video)
    _dilate_e_base = deltaE_cie76(_rgb_full_video, np.zeros_like(_rgb_full_video))
    _dilate_e_relative = _dilate_e / (_dilate_e_base + 1e-6)
    _dilate_e_threshold = threshold_otsu(_dilate_e)
    _dilate_e_relative_threshold = threshold_otsu(_dilate_e_relative)


    for (rgb_full_path, rgb_removed_path, mask_path) in zip(rgb_full_paths, rgb_removed_paths, mask_paths):
        rgb_full = media.read_image(rgb_full_path)
        rgb_removed = media.read_image(rgb_removed_path)
        mask = media.read_image(mask_path)

        # ----------------------------------- mask ----------------------------------- #
        # mask = np.where(np.logical_and(mask >63, mask < 192), 127, mask)
        # mask = np.where(mask >= 192, 255, mask)
        # mask = np.where(mask <= 63, 0, mask)
        # # ! remap color (fg: black, bg: white)
        # mask = np.where(mask == 255, 127, mask) # remove other foreground, map them to gray
        # mask = np.where(mask == 127, 255, mask) # background to white
        mask = np.zeros_like(mask)
        if len(mask.shape) == 2:
            mask = np.stack([mask, mask, mask], axis=-1)

        rgb_full_lab = rgb2lab(rgb_full)
        rgb_removed_lab = rgb2lab(rgb_removed)
        delta_e = deltaE_cie76(rgb_full_lab, rgb_removed_lab)

        # Relative delta: normalize by perceptual magnitude
        delta_e_base = deltaE_cie76(rgb_full_lab, np.zeros_like(rgb_full_lab))
        delta_e_relative = delta_e / (delta_e_base + 1e-6)
        relative_mask = threshold_mask(delta_e_relative, dilate_radius=dilate_radius, otsu_weight=otsu_weight, threshold=_dilate_e_relative_threshold)

        # Absolute delta
        absolute_mask = threshold_mask(delta_e, dilate_radius=dilate_radius, otsu_weight=otsu_weight, threshold=_dilate_e_threshold)

        # merge relative and absolute delta
        merge_mask = np.logical_or(absolute_mask, relative_mask)
        merge_mask = threshold_mask(merge_mask, dilate_radius=dilate_radius, otsu_weight=otsu_weight)
        # keep the N largest components
        N_largest_components = 40
        min_size_threshold = 20
        labeled_mask, num_labels = label(merge_mask, return_num=True)
        if num_labels > 0:
            component_sizes = np.bincount(labeled_mask.ravel())[1:] # ignore background (label 0)
            original_labels = np.arange(1, num_labels + 1)
            large_enough_indices = np.where(component_sizes >= min_size_threshold)[0]
            if len(large_enough_indices) > 0:
                candidate_labels = original_labels[large_enough_indices]
                candidate_sizes = component_sizes[large_enough_indices]
                num_to_keep = min(N_largest_components, len(candidate_labels))
                largest_indices = np.argsort(candidate_sizes)[-num_to_keep:]
                final_labels = candidate_labels[largest_indices]
                merge_mask = np.isin(labeled_mask, final_labels)
            else:
                merge_mask = np.zeros_like(merge_mask, dtype=bool)
        # dilate
        merge_mask = binary_dilation(merge_mask, footprint=disk(5))

        # ! remap color (effect: gray)
        # merge_mask = np.where(merge_mask, 127, 0)
        # merge_mask = np.stack([merge_mask, merge_mask, merge_mask], axis=-1).astype(np.uint8)
        # final_mask = np.where(merge_mask == 127, 127,mask)
        final_mask = merge_mask
        # copy the rgb and save mask
        save_name = os.path.basename(mask_path)
        save_path = os.path.join(out_parent_dir, 'trimask', save_name)
        media.write_image(save_path, final_mask)

        # shutil.copy(rgb_full_path, os.path.join(out_parent_dir, 'input', os.path.basename(rgb_full_path)))
        # shutil.copy(rgb_removed_path, os.path.join(out_parent_dir, 'bg', os.path.basename(rgb_removed_path)))
    print(f'Finished {dir_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='omnimatte', choices=['kubric', 'omnimatte', 'rose', 'omnimatte_zero', 'movie'])
    args = parser.parse_args()
    if args.mode == 'kubric':
        n_process = max(cpu_count() - 1, 1)
        in_dir = '/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/train-casper/kubric_ours'
        out_dir = '/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/train-casper/kubric'
        file_list = [file for file in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, file))]
        with Pool(processes=n_process) as pool:
            pool.starmap(create_kubric_mask, [(in_dir, out_dir, file) for file in file_list])
    elif args.mode == 'omnimatte':
        n_process = max(cpu_count() - 1, 1)
        in_dir = '/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/train-casper/omnimatte+tripod_ours'
        out_dir = '/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/train-casper/omnimatte+tripod'
        file_list = [file for file in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, file))]
        with Pool(processes=n_process) as pool:
            pool.starmap(create_omnimatte_mask, [(in_dir, out_dir, file) for file in file_list])
    elif args.mode == 'movie':
        # n_process = max(cpu_count() - 1, 1)
        n_process = 4
        in_dir = '/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/train-casper/movie'
        out_dir = '/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/train-casper/movie'
        file_list = [file for file in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, file))]
        with Pool(processes=n_process) as pool:
            pool.starmap(create_movie_mask, [(in_dir, out_dir, file) for file in file_list])
    elif args.mode == 'rose':
        # n_process = max(cpu_count() - 1, 1)
        n_process = 11 # ! requires large memory ~500GB RAM 
        n_process = 6
        in_dir = '/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/train-casper/rose_ours'
        out_dir = '/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/train-casper/rose'
        file_list = [file for file in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, file))]
        print(f'Processing {len(file_list)} files')
        with Pool(processes=n_process) as pool:
            pool.starmap(create_rose_mask, [(in_dir, out_dir, file) for file in file_list])
            # pool.starmap(create_rose_mask, [(in_dir, out_dir, file) for file in file_list][::-1])

        def _video_meta(path):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release(); return None
            meta = {
                'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': float(cap.get(cv2.CAP_PROP_FPS)),
                'w': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'h': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
            }
            cap.release()
            if meta['frames'] <= 0 or meta['fps'] <= 0 or meta['w'] <= 0 or meta['h'] <= 0:
                return None
            return meta

        def validate_rose_outputs_specs(out_dir, fps_tol=1e-3, frame_tol=0):
            invalid = []
            for name in os.listdir(out_dir):
                d = os.path.join(out_dir, name)
                if not os.path.isdir(d): continue

                full = os.path.join(d, 'rgb_full.mp4')
                rem  = os.path.join(d, 'rgb_removed.mp4')
                mask = os.path.join(d, 'mask.mp4')
                if not (os.path.exists(full) and os.path.exists(rem) and os.path.exists(mask)):
                    invalid.append((d, 'missing file')); continue

                m_full = _video_meta(full)
                m_rem  = _video_meta(rem)
                m_mask = _video_meta(mask)
                if m_full is None or m_rem is None or m_mask is None:
                    invalid.append((d, 'bad meta')); continue

                # same dimensions
                if not (m_full['w'] == m_rem['w'] == m_mask['w'] and m_full['h'] == m_rem['h'] == m_mask['h']):
                    invalid.append((d, 'dim mismatch')); continue

                # fps close
                if not (abs(m_full['fps'] - m_rem['fps']) <= fps_tol and abs(m_full['fps'] - m_mask['fps']) <= fps_tol):
                    invalid.append((d, 'fps mismatch')); continue

                # same frame counts (allow small tolerance if needed)
                if not (abs(m_full['frames'] - m_rem['frames']) <= frame_tol and abs(m_full['frames'] - m_mask['frames']) <= frame_tol):
                    invalid.append((d, 'frame count mismatch')); continue

                # optional: same codec for rgb pair
                # if m_full['fourcc'] != m_rem['fourcc']: invalid.append((d, 'codec mismatch')); continue

            print(f'Invalid outputs: {len(invalid)}')
            if invalid[:5]: print('Examples:', invalid[:5])
            return invalid
        invalid = validate_rose_outputs_specs(out_dir)

    elif args.mode == 'omnimatte_zero':
        n_process = max(cpu_count() - 1, 1)
        in_dir = '/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/train-casper/omnimatte_zero'
        out_dir = '/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/train-casper/omnimatte_zero'
        file_list = [file for file in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, file))]
        with Pool(processes=n_process) as pool:
            pool.starmap(create_omnimatte_zero_mask, [(in_dir, out_dir, file) for file in file_list])
    else:
        raise ValueError(f'Invalid mode: {args.mode}')