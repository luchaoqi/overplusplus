# code to run inference over multiple gpus
# single-gpu cmd template:
# python inference/cogvideox_fun/predict_v2v.py \
#     --config.experiment.save_foreground=True \
#     --config.experiment.save_path="output_temp" \
#     --config.data.data_rootdir="examples" \
#     --config.experiment.run_seqs="truck-water" \
#     --config.experiment.skip_if_exists=False


import os
import sys
import subprocess
import signal
import time
import argparse

def iterate_over_dirs(root_dir):
    for dir_name in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, dir_name)):
            dir_path = os.path.join(root_dir, dir_name)
            yield dir_name


# def signal_handler(signum, frame):
#     """Handle Ctrl+C and other signals by terminating all child processes"""
#     print(f"\nReceived signal {signum}. Terminating all subprocesses...")
    
#     for process in process_list:
#         if process.poll() is None:  # Process is still running
#             try:
#                 # Try to terminate gracefully first
#                 process.terminate()
#                 print(f"Sent TERM signal to process {process.pid}")
#             except Exception as e:
#                 print(f"Error terminating process {process.pid}: {e}")
    
#     # Give processes a moment to terminate gracefully
#     time.sleep(2)
    
#     # Force kill any remaining processes
#     for process in process_list:
#         if process.poll() is None:  # Process is still running
#             try:
#                 process.kill()
#                 print(f"Force killed process {process.pid}")
#             except Exception as e:
#                 print(f"Error killing process {process.pid}: {e}")
    
#     print("All subprocesses terminated. Exiting...")
#     sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on multiple GPUs')
    parser.add_argument('--output_dir', type=str, 
                       default="/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/output_temp",
                       help='Directory to save output files')
    parser.add_argument('--input_dir', type=str,
                       default="/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/examples",
                       help='Root directory containing input data')
    parser.add_argument('--n_gpus', type=int, default=4,
                       help='Number of GPUs to use for inference')
    parser.add_argument('--mode', type=str,
                        default="cogvideox_fun",
                        help='Name of the model to use for inference, e.g. "cogvideox_fun" or "wan2.1_fun"')
    parser.add_argument('--transformer_path', type=str,
                        default=None,
                        help='Path to transformer model')
    parser.add_argument('--dilate_width', type=int,
                        default=0,
                        help='Dilate width for the mask')
    parser.add_argument('--skip_if_exists', type=bool, default=False,
                        help='Whether to skip if the output already exists')
    # parser.add_argument('--use_trimask', action='store_true',
    #                     help='Whether to use trimask')
    parser.add_argument('--prompt_guidance_scale', type=float,
                        default=1.0,
                        help='Guidance scale for the model')
    parser.add_argument('--mask_guidance_scale', type=float,
                        default=0.0,
                        help='Guidance scale for the model')
    args = parser.parse_args()
    output_dir = args.output_dir
    data_rootdir = args.input_dir
    n_gpus = args.n_gpus
    mode = args.mode
    transformer_path = args.transformer_path
    dilate_width = args.dilate_width
    # use_trimask = args.use_trimask
    skip_if_exists = args.skip_if_exists
    prompt_guidance_scale = args.prompt_guidance_scale
    mask_guidance_scale = args.mask_guidance_scale
    # # Set up signal handlers for graceful termination
    # signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    # signal.signal(signal.SIGTERM, signal_handler)  # Kill command
    
    # print(f"Starting inference on {n_gpus} GPUs. Press Ctrl+C to terminate all processes.")
    
    # # * current implementation needs to load/offload pipeline for each gpu - this is not the final version
    # dir_name_list = sorted(list(iterate_over_dirs(data_rootdir)))
    # for i in range(0, len(dir_name_list), n_gpus):
    #     process_list = []
    #     for gpu_id in range(n_gpus):
    #         dir_name = dir_name_list[i+gpu_id]
    #         # * im using matting mode "clean_bg" for now to trigger fg_id=-1 only
    #         code_to_run = f"""
    #         CUDA_VISIBLE_DEVICES={gpu_id} python inference/cogvideox_fun/predict_v2v.py \
    #         --config.experiment.save_path="{output_dir}" \
    #         --config.data.data_rootdir="{data_rootdir}" \
    #         --config.experiment.run_seqs="{dir_name}" \
    #         --config.experiment.skip_if_exists=False \
    #         --config.experiment.save_foreground=True
    #         """
    #         print(code_to_run)
    #         process_list.append(subprocess.Popen(code_to_run, shell=True))
    #     for process in process_list:
    #         process.wait()
    # exit()
    data_rootdir_list = data_rootdir.split(",")
    for data_rootdir in data_rootdir_list:    
        print(f"Running inference for {data_rootdir}")

        dir_name_list = sorted(list(iterate_over_dirs(data_rootdir)))
        # Group directories by GPU
        gpu_groups = [[] for _ in range(n_gpus)]
        for idx, dir_name in enumerate(dir_name_list):
            gpu_id = idx % n_gpus
            gpu_groups[gpu_id].append(dir_name)

        process_list = []
        for gpu_id in range(n_gpus):
            dir_name_list_i_str = ",".join(gpu_groups[gpu_id])
            if dir_name_list_i_str == "":
                continue
            # * add transformer path if provided, otherwise use default from the config file
            transformer_arg = ""
            trimask_arg = ""
            if transformer_path is not None:
                transformer_arg = f'--config.video_model.transformer_path="{transformer_path}"'
            # if not use_trimask:
            #     trimask_arg = "--config.video_model.use_trimask=False"

            # ! config.data.dilate_width=0 is important for avoiding hallucinated effects beyond the mask
            if mode == "cogvideox_fun":
                code_to_run = f"""
                    CUDA_VISIBLE_DEVICES={gpu_id} python inference/cogvideox_fun/predict_v2v.py \
                    --config.experiment.save_path="{output_dir}" \
                    --config.data.data_rootdir="{data_rootdir}" \
                    --config.experiment.run_seqs="{dir_name_list_i_str}" \
                    --config.experiment.skip_if_exists={skip_if_exists} \
                    --config.experiment.save_foreground=True \
                    --config.data.dilate_width={dilate_width} \
                    {transformer_arg} \
                    --config.video_model.guidance_scale={prompt_guidance_scale} \
                    --config.video_model.mask_guidance_scale={mask_guidance_scale}
                """
            elif mode == "wan2.1_fun":
                code_to_run = f"""
                CUDA_VISIBLE_DEVICES={gpu_id} python inference/wan2.1_fun/predict_v2v.py \
                --config.experiment.save_path="{output_dir}" \
                --config.data.data_rootdir="{data_rootdir}" \
                --config.experiment.run_seqs="{dir_name_list_i_str}" \
                --config.experiment.skip_if_exists={skip_if_exists} \
                --config.data.dilate_width={dilate_width} \
                {trimask_arg} \
                {transformer_arg}
                """
            elif mode == "original":
                # * for original cogvideox, we load from config file
                code_to_run = f"""
                    CUDA_VISIBLE_DEVICES={gpu_id} python inference/cogvideox_fun/predict_v2v.py \
                    --config.experiment.save_path="{output_dir}" \
                    --config.data.data_rootdir="{data_rootdir}" \
                    --config.experiment.run_seqs="{dir_name_list_i_str}" \
                    --config.experiment.skip_if_exists={skip_if_exists} \
                    --config.experiment.save_foreground=True \
                    --config.data.dilate_width={dilate_width}
                """
            elif mode == "t2v":
                #  * use default cfg guidance scale=6 for t2v mode
                code_to_run = f"""
                    CUDA_VISIBLE_DEVICES={gpu_id} python inference/cogvideox_fun/predict_v2v_prompt.py \
                    --config.experiment.save_path="{output_dir}" \
                    --config.experiment.skip_if_exists={skip_if_exists} \
                    --config.experiment.save_foreground=True \
                    --config.data.dilate_width={dilate_width} \
                    --config.video_model.guidance_scale=6 \
                    --config.experiment.run_seqs="{gpu_id + 1}/{n_gpus}" \
                    --config.video_model.transformer_path="/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP/transformer/diffusion_pytorch_model.safetensors"
                """
                # --config.video_model.transformer_path="/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP/transformer/diffusion_pytorch_model.safetensors"
                # --config.video_model.transformer_path="/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/training/cogvideox-casper-gen_omnimatte_dualmask_9/checkpoint-1000/transformer/diffusion_pytorch_model.safetensors"
                
            print(code_to_run)
            process_list.append(subprocess.Popen(code_to_run, shell=True))
        for process in process_list:
            process.wait()
    # exit()