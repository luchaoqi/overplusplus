# copied from https://github.com/brown-palm/force-prompting/blob/main/scripts/test_dataset_preprocessing/point_force/app_dataset_preprocessing.py

import os
import io
import base64
import cv2
import json
import csv
import copy
import numpy as np
from PIL import Image
from openai import OpenAI
from multiprocessing import Pool
import multiprocessing as mp
import shutil



def optimize_prompt(caption):
    # input_video_prompt = caption['short_prompt']
    input_video_prompt = caption['prompt']

    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        SYSTEM_PROMPT = """
You are an expert video captioner specializing in visual effects (VFX).  
Your task is to generate multiple highly descriptive captions that augment an input caption with **different variations of the same effect type**.  

### Rules:
1. **Identify the effect type** in the input (e.g., splash, smoke, shadow, reflection, fire, dust, ripple, etc.).  
2. **Keep the scene, subjects, and actions the same**. Do not replace or remove them.  
3. Only change the **effect style**, varying attributes such as:  
   - Color (e.g., gray smoke → bright red smoke)  
   - Intensity (subtle → violent)  
   - Spatial spread (localized → wide, global)  
   - Texture / density (thin wispy vs thick heavy)  
4. Each output caption must be **similar in length** to the input caption.  
5. Language should be **natural, vivid, and concise**.  
6. Provide **N diverse variations** per input (N is specified by the user, use N=3 if not specified). The variations should be as much diverse as possible. 

### Output Format:
Return the captions as a JSON list of strings, like this:  
[
  "caption variation 1",
  "caption variation 2",
  "caption variation 3"
]

### Example:
Input: "A woman is surfing on a beach, leaving a long turbulent wake."
Output:
[
  "A woman is surfing on a beach, leaving a mild gentle ripple behind her board.",
  "A woman is surfing on a beach, trailing a thin spray of water that glitters in the sunlight.",
  "A woman is surfing on a beach, with a violent splash crashing outward in all directions."
]
Input: "A car is driving on a highway, leaving shadows on the ground."
Output:
[
  "A car is driving on a highway, casting soft diffuse shadows with blurry edges and low contrast that blend gently into the ground.",
  "A car is driving on a highway, casting harsh shadows with sharp edges and high contrast that fall crisply across the ground.",
]
"""     
        # Call the API with image and prompt
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": f"{SYSTEM_PROMPT}"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_video_prompt},
                    ],
                },
            ],
            # temperature=0.01,
            # top_p=0.7,
            # stream=False,
            # max_tokens=250,
        )
        
        optimized_prompt = response.choices[0].message.content
        # * formatting the output
        optimized_prompt = json.loads(optimized_prompt)
        n = len(optimized_prompt)
        captions = []
        for i in range(n):
            new_caption = copy.deepcopy(caption)
            new_caption['path'] = caption['path'] + f"-{i:04d}"
            # TODO
            # new_caption['short_prompt'] = optimized_prompt[i]
            new_caption['prompt'] = optimized_prompt[i]
            captions.append(new_caption)
        return captions
    except Exception as e:
        print(e)
        return caption

if __name__ == '__main__':
    captions = json.load(open('/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/train-casper/omnimatte+tripod.json'))
    # captions = captions[:2]
    with Pool(processes=mp.cpu_count() // 2) as pool:
        results = pool.map(optimize_prompt, captions)
    # formatting and unlisting the results
    results = [item for sublist in results for item in sublist]
    target_path = '/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/train-casper/omnimatte+tripod_augment.json'
    json.dump(results, open(target_path, 'w'), indent=4)