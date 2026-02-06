# codes from https://gist.github.com/a-r-r-o-w/4dee20250e82f4e44690a02351324a4a

import argparse
import os
import glob
import json
import random
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import transformers
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance_data_root", type=str, required=True, help="Base folder where video files are located."
    )
    parser.add_argument("--cache_dir", type=str, default="~/.cache", help="Path to where models are stored.")
    parser.add_argument(
        "--output_path", type=str, default="captions.json", help="File path where dataset csv should be stored."
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=226, help="Maximum number of new tokens to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    return parser.parse_args()

SYSTEM_PROMPT = """
You are part of a team that creates videos using generative models. Your task is to write prompts that guide the model to generate realistic videos grounded in physical and environmental detail, especially those caused by **interactions between the foreground and background**.

Your descriptions must:
- Highlight **environmental effects** (e.g., light, shadow, water, dust, wind, smoke, reflections).
- Emphasize effects **caused by interaction** — such as splashes from a foot hitting water, dust clouds from wheels skidding on dirt, or shadows cast from a moving object onto a textured surface.
- Specify the **degree**, **intensity**, and **spatial spread** of those effects — such as whether the splash is subtle or turbulent, whether shadows are soft or harsh, whether smoke is thin or dense, whether reflections are cloudy or glossy.
- Use precise, physical language (e.g., "fine mist hangs in the air", "glare flares sharply off wet asphalt", "ripples cascade outward in concentric rings").
- Focus on **visual physics**, **perceptual realism**, and **spatio-temporal interactions**.

Only generate **one vivid and specific prompt per request**.

Example prompts:
- "A child jumps into a shallow muddy puddle, sending thick arcs of dirty water in all directions. Ripples cascade outward, colliding with fallen leaves. The splash leaves droplets on nearby bricks, which slowly drip down."
- "A race car speeds across a dry desert track, sending up dense clouds of dust that billow and linger in the air. The dust trails stretch for meters, gradually dispersing into the golden light."
- "A skateboarder grinds a wet ledge in a dimly lit alley, causing bursts of water spray and streaks of reflections to shimmer under passing headlights. Drops scatter along the ground and slowly fade."
""".strip()


USER_PROMPT = """
Given the following summary, generate a vivid, physics-aware prompt for a video generation model. Focus only on the following effect: {0}, describe it in detail without mentioning other environmental effects.

Your description should:
- Emphasize the **intensity**, **spatial spread**, and **temporal dynamics** of the {0}.
- Specify the **degree**, **strength**, and **spatial reach** of the {0}. For example:
  - Does dust linger or dissipate quickly?
  - Does water splash violently or ripple gently?
  - Are shadows soft and shallow, or sharp and dark, depending on light sources?
- Use **sensory and physical language** that reflects real-world dynamics.

Summary:
```
{1}
```
Please limit the prompt to [{2}] words.
""".strip()


QUESTION = """
Describe the video in detail, with a special focus on environmental effects and how they interact with the scene.

Pay close attention to:
- The **type and intensity** of physical effects — such as soft vs. harsh lighting, subtle vs. violent splashes, thin vs. dense smoke, or gentle vs. strong wind.
- How these effects **interact with the foreground and background** — e.g., shadows cast across surfaces, splashes disrupting water, smoke enveloping objects, or reflections shifting with movement.
- The **spatial extent** (how far the effects spread) and **temporal behavior** (how quickly they appear, move, or dissipate).

Avoid generic summaries. Focus on how **physical forces affect materials** and **how visible cues evolve over time**.
""".strip()

IDENTIFY_DOMINANT_EFFECT = """
Given the video, return the **single most dominant environmental or physical interaction effect** from the following list:

- water
- dust
- shadow
- reflection
- smoke

Respond with **only one keyword**, exactly as written above — no explanation or extra text.

If no clear effect is visible, but there is still interaction (e.g., motion, contact), respond with: foreground-background interaction.
""".strip()

MAX_NUM_FRAMES = 49 * 2


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_video(video_path: str):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    if os.path.isdir(os.path.join(video_path, 'bg')):
        # frames = [Image.open(f).resize((256, 128), resample=Image.LANCZOS)
        #           for f in sorted(glob.glob(os.path.join(video_path, 'bg', '*.png')))][:MAX_NUM_FRAMES]
        frame_paths = sorted(
            glob.glob(os.path.join(video_path, 'bg', '*.png')) +
            glob.glob(os.path.join(video_path, 'bg', '*.jpg'))
        )
        frames = [Image.open(f).resize((256, 128), resample=Image.LANCZOS) for f in frame_paths][:MAX_NUM_FRAMES]
    else:
        video_path = os.path.join(video_path, 'rgb_removed.mp4')
        # video_path = os.path.join(video_path, 'input_video.mp4')
        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = 2
        frame_idx = [i for i in range(0, len(vr), sample_fps)][:MAX_NUM_FRAMES]
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")).resize((256, 128), resample=Image.LANCZOS) for v in frames]

    print("num frames:", len(frames))
    return frames

def encode_video_negative(video_path: str):
    # TODO: implement this
    pass

@torch.no_grad()
def main(args: Dict[str, Any]):
    set_seed(args.seed)

    video_files = [file for file in os.listdir(args.instance_data_root)]
    video_files = sorted([os.path.join(args.instance_data_root, file) for file in video_files])
    if args.end_index is not None:
        print(f'[INFO] processing {args.start_index} to {args.end_index}')
        video_files = video_files[args.start_index:args.end_index]
    video_descriptions = {}

    model_id = "openbmb/MiniCPM-V-2_6"
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    for filepath in video_files:
        print(f"Generating video summary for file: `{filepath}`")
        frames = encode_video(filepath)
        msgs = [{"role": "user", "content": frames + [QUESTION]}]
        msgs_effect = [{"role": "user", "content": frames + [IDENTIFY_DOMINANT_EFFECT]}]

        params = {
            "use_image_id": False,
            "max_slice_nums": 10,
        }

        description = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, **params)
        dominant_effect = model.chat(image=None, msgs=msgs_effect, tokenizer=tokenizer, **params)
        video_descriptions[filepath] = {
            "summary": description,
            "dominant_effect": dominant_effect,
        }

    del model
    del tokenizer
    model = None
    tokenizer = None
    torch.cuda.empty_cache()

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        # model_kwargs={
        #     "local_files_only": True,
        #     "torch_dtype": torch.bfloat16,
        # },
    )

    for filepath in video_files:
        print(f"Generating captions for file: `{filepath}`")
        summary = video_descriptions[filepath]["summary"]
        dominant_effect = video_descriptions[filepath]["dominant_effect"]

        for prompt_type, num_words in [("short_prompt", 25), ("prompt", 75), ("verbose_prompt", 125)]:
            # user_prompt = USER_PROMPT.format(summary, num_words)
            user_prompt = USER_PROMPT.format(dominant_effect, summary, num_words)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=226,
            )
            video_descriptions[filepath][prompt_type] = outputs[0]["generated_text"][-1]["content"]

    # for filepath in negative_video_files:
    #         # now generate the negative prompt
    #         user_prompt = NEGATIVE_USER_PROMPT.format(video_descriptions[filepath]["summary"], num_words)
    #         messages = [
    #             {"role": "system", "content": NEGATIVE_SYSTEM_PROMPT},
    #             {"role": "user", "content": user_prompt},
    #         ]
    #         outputs = pipeline(
    #             messages,
    #             max_new_tokens=226,
    #         )
    #         video_descriptions[filepath][f"negative_{prompt_type}"] = outputs[0]["generated_text"][-1]["content"]

    df_data = []
    for filepath, description in video_descriptions.items():
        relative_path = os.path.relpath(filepath, args.instance_data_root)
        df_data.append(
            {
                "path": relative_path,
                "short_prompt": description.get("short_prompt", ""),
                "prompt": description.get("prompt", ""),
                "verbose_prompt": description.get("verbose_prompt", ""),
                "summary": description.get("summary", ""),
                "dominant_effect": description.get("dominant_effect", ""),
                # "negative_short_prompt": description.get("negative_short_prompt", ""),
                # "negative_prompt": description.get("negative_prompt", ""),
                # "negative_verbose_prompt": description.get("negative_verbose_prompt", ""),
            }
        )

    json.dump(df_data, open(args.output_path, "w"), indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args)

    # Example usage:
    # python3 workflows/caption_videos.py \
    #   --instance_data_root <PATH_TO_DATA> \
    #   --cache_dir <PATH_TO_MODEL_CACHE_DIR> \
    #   --output_path <PATH_TO_DATASET_SAVEDIR>
