import argparse
import glob
import json
import multiprocessing
import os
import subprocess
import sys
import shutil


def run_captioning_worker(
    gpu_id: int,
    instance_data_root: str,
    cache_dir: str,
    output_path: str,
    start_index: int,
    end_index: int,
):
    """
    A worker function that invokes the caption_videos.py script for a subset of the data.
    It sets CUDA_VISIBLE_DEVICES to assign the process to a specific GPU.
    """
    print(
        f"Starting process on GPU {gpu_id} for videos from index {start_index} to {end_index}..."
    )

    # Prepare the command to run the original script
    command = [
        sys.executable,  # Use the same python interpreter that runs this script
        "caption_videos.py",
        "--instance_data_root",
        instance_data_root,
        "--cache_dir",
        cache_dir,
        "--output_path",
        output_path,
        "--start_index",
        str(start_index),
        "--end_index",
        str(end_index),
    ]

    # Set the CUDA_VISIBLE_DEVICES environment variable for the subprocess
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Launch the subprocess
    try:
        subprocess.run(
            command, check=True, env=env, text=True, capture_output=True
        )
        print(f"Successfully finished process on GPU {gpu_id}.")
    except subprocess.CalledProcessError as e:
        print(f"Error in subprocess on GPU {gpu_id}:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise e


def merge_json_files(input_dir: str, output_path: str):
    """Merges all JSON files in a directory into a single JSON file."""
    all_data = []
    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    for file_path in json_files:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    print(
                        f"Warning: {file_path} does not contain a list. Skipping."
                    )
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not decode JSON from {file_path}. Skipping."
                )

    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=4)

    print(
        f"Merged {len(json_files)} files into {output_path}. Total records: {len(all_data)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run video captioning in parallel across multiple GPUs."
    )
    parser.add_argument(
        "--instance_data_root",
        type=str,
        required=True,
        help="Base folder where video files are located.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/captions/cache",
        help="Path to where models are stored.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/captions/caption_outputs",
        help="Directory to store partial JSON results.",
    )
    parser.add_argument(
        "--final_output_path",
        type=str,
        default="/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/captions/captions.json",
        help="Path for the final merged JSON file.",
    )
    args = parser.parse_args()

    # Prepare output directory
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    # args.cache_dir = os.path.expanduser(args.cache_dir)
    # clean cache dir
    if os.path.exists(args.cache_dir):
        shutil.rmtree(args.cache_dir)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Get list of GPUs
    gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(",")]
    num_gpus = len(gpu_ids)
    if num_gpus == 0:
        print("Error: No GPUs specified.")
        return

    # Get list of videos
    video_files = [f for f in os.listdir(args.instance_data_root) if os.path.isdir(os.path.join(args.instance_data_root, f))]
    total_videos = len(video_files)

    # Calculate work distribution
    chunk_size = (total_videos + num_gpus - 1) // num_gpus
    print(f"Total videos: {total_videos}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Chunk size per GPU: {chunk_size}")

    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, total_videos)

        if start_index >= total_videos:
            continue

        output_path = os.path.join(
            args.output_dir, f"captions_{start_index}_{end_index}.json"
        )
        process_args = (
            gpu_id,
            args.instance_data_root,
            args.cache_dir,
            output_path,
            start_index,
            end_index,
        )

        p = multiprocessing.Process(
            target=run_captioning_worker, args=process_args
        )
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("\nAll captioning jobs finished. Merging results...")
    merge_json_files(args.output_dir, args.final_output_path)


if __name__ == "__main__":
    # Set start method to 'spawn' for CUDA compatibility in multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()