import json
import os
import shutil
from pathlib import Path
from config import RUNS_DIR, FAVORITES_FILE, STARS_FILE
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Collect favorite images into a single folder.")
parser.add_argument('destination_folder', type=str, help='Destination folder name inside gallery/favorites/')
parser.add_argument('stars_or_hearts', type=str, choices=['stars', 'hearts'], default='hearts', help='Whether to collect starred images or hearted images')
args = parser.parse_args()
dest_folder = args.destination_folder
stars_or_hearts = args.stars_or_hearts

if stars_or_hearts not in ['stars', 'hearts']:
   raise ValueError("stars_or_hearts must be either 'stars' or 'hearts'")
if not dest_folder:
   raise ValueError("destination_folder must be a valid folder name")

# Get the directory of this script (gallery folder)
script_dir = Path(__file__).parent
# Load favorites from config location
favs = json.load(open(FAVORITES_FILE if stars_or_hearts == 'hearts' else STARS_FILE))

images = favs['images']

def collect_images_to(dest_folder: Path) -> None:
   for image in images:
      # Use configured runs directory
      image_path = RUNS_DIR / image
      # copy that image to favorites folder
      destination_folder = script_dir / 'favorites' / dest_folder
      # make sure the folder exists
      destination_folder.mkdir(parents=True, exist_ok=True)

      settings_path = image_path.parent / 'settings.json'
      if settings_path.exists():
         # load up the json into a dictionary
         settings = json.load(open(settings_path))
         prompts = settings.get('prompts', {})
         reference_image_path = prompts.get('reference_image_path', None) if prompts else None

         # print("Checking if reference image path exists: ", reference_image_path)
         if reference_image_path:
            print("Found reference image path: ", reference_image_path)
            # copy the reference image too
            ref_image_full_path = RUNS_DIR / reference_image_path
            destination_folder = destination_folder / 'image_image' / ref_image_full_path.stem 
            # Ensure the directory exists before writing
            destination_folder.mkdir(parents=True, exist_ok=True)
            reference_file_path = destination_folder / os.path.basename(reference_image_path)
            if not reference_file_path.exists():
               shutil.copy(ref_image_full_path, reference_file_path)
            destination_folder = destination_folder / settings.get('clip_model', {}).get('name', 'unknown_model')
            destination_folder.mkdir(parents=True, exist_ok=True)

         else:
            # print("No reference image path found in prompts.")
            # See if we can get the prompt
            initial_prompt = prompts.get('clip_initial_text_prompt', None)
            target_prompt = prompts.get('clip_target_text_prompt', None)
            if initial_prompt and target_prompt:
               destination_folder = destination_folder / 'directional_text_prompt' / target_prompt.replace(" ", "_")[:50]
               # Ensure the directory exists before writing
               destination_folder.mkdir(parents=True, exist_ok=True)
               # Write the target and initial prompt to a text file
               with open(destination_folder / 'prompts.txt', 'w') as f:
                  f.write(f"Initial Prompt: {initial_prompt}\n")
                  f.write(f"Target Prompt: {target_prompt}\n")

      if image_path.exists():
         shutil.copy(image_path, destination_folder / os.path.basename(image))

if __name__ == "__main__":
   collect_images_to(dest_folder)