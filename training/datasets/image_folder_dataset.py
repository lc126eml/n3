import os
import os.path as osp
import glob
from PIL import Image
import numpy as np
import cv2

from datasets.base.base_multiview_dataset import BaseMultiViewDataset
from fast3r.viz.video_utils import extract_frames_from_video


class ImageFolderDataset(BaseMultiViewDataset):
    """
    A dataset that loads images from a directory of subfolders (scenes) or video files.
    It does not require ground truth depth or camera poses, making it suitable for inference
    or for training methods that do not require such supervision.
    """
    def __init__(self, *args, ROOT, samples_per_scene=1, max_interval=10, **kwargs):
        self.ROOT = ROOT
        self.video = True  # Treat image sequences as video-like
        self.is_metric = False  # No ground truth depth, so not metric.
        self.samples_per_scene = samples_per_scene
        self.max_interval = max_interval
        
        # We are not using ground truth, so we can set n_corres to 0
        kwargs['n_corres'] = 0 
        
        super().__init__(*args, **kwargs)
        self._load_scenes()

    def _load_scenes(self):
        """
        Recursively scans the root directory for scenes. A scene is defined as either:
        1. A directory that directly contains image files (and no videos).
        2. A video file.
        The scan is recursive. If a directory contains videos, it is not treated as an
        image-folder scene, and its subdirectories are not scanned.
        """
        self.scenes = []
        self.scene_data = {}  # Maps a relative scene path/name to a list of image paths

        if not osp.isdir(self.ROOT):
            raise FileNotFoundError(f"Root directory not found: {self.ROOT}")

        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        image_extensions = ('.png', '.jpg', '.jpeg')

        for dirpath, dirnames, filenames in os.walk(self.ROOT, topdown=True):
            if '.tmp_frames' in dirpath.split(os.sep):
                dirnames[:] = []  # Don't descend into .tmp_frames
                continue

            video_files = [f for f in filenames if f.lower().endswith(video_extensions)]
            
            # If videos are present, treat each as a scene and stop descending.
            if video_files:
                dirnames[:] = []  # Stop recursion for this branch
                for video_file in video_files:
                    video_path = osp.join(dirpath, video_file)
                    scene_name = osp.relpath(video_path, self.ROOT)
                    
                    temp_frame_dir = osp.join(self.ROOT, '.tmp_frames', scene_name)
                    os.makedirs(temp_frame_dir, exist_ok=True)
                    
                    extracted_frames = extract_frames_from_video(video_path, temp_frame_dir)
                    
                    self.scenes.append(scene_name)
                    self.scene_data[scene_name] = extracted_frames
                continue # Move to the next directory in os.walk

            # If no videos, check for an image-folder scene.
            image_files = [f for f in filenames if f.lower().endswith(image_extensions)]
            if image_files:
                dirnames[:] = []  # Stop recursion, this is a scene.
                
                full_image_paths = [osp.join(dirpath, f) for f in image_files]
                
                scene_name = osp.relpath(dirpath, self.ROOT)
                if scene_name == '.':
                    scene_name = osp.basename(self.ROOT)
                
                self.scenes.append(scene_name)
                self.scene_data[scene_name] = sorted(full_image_paths)

        if not self.scenes:
            raise ValueError(f"No valid scenes with at least {self.num_views} images/frames found in {self.ROOT}")

        self.scenes.sort()

    def __len__(self):
        return len(self.scenes) * self.samples_per_scene

    def _get_views(self, idx, resolution, rng, num_views):

        scene_idx = idx // self.samples_per_scene
        scene_name = self.scenes[scene_idx]
        image_paths = self.scene_data[scene_name]

        if num_views <= len(image_paths):
            # Use the requested method to sample `num_views` indices from the image list
            pos, ordered_video = self.efficient_random_intervals_revised(
                start=0, max_idx=len(image_paths), num_elements=num_views, rng=rng, min_interval=1, max_interval=self.max_interval
            )
            
            selected_image_paths = [image_paths[i] for i in pos]
        else:
            selected_image_paths = image_paths
            ordered_video = True

        views = []
        for i, img_path in enumerate(selected_image_paths):
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Warning: Could not load image {img_path}. Skipping view. Error: {e}")
                continue

            # --- Preprocessing aligned with `eval_manual.py` (`load_images`) ---
            # 1. Resize so the image covers the target resolution, preserving aspect ratio.
            target_W, target_H = resolution
            original_W, original_H = img.size
            scale = max(target_W / original_W, target_H / original_H)
            
            resample_filter = Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.BICUBIC
            resized_W, resized_H = int(round(original_W * scale)), int(round(original_H * scale))
            img = img.resize((resized_W, resized_H), resample=resample_filter)

            # 2. Center crop to the exact target resolution.
            left = (resized_W - resolution[0]) // 2
            top = (resized_H - resolution[1]) // 2
            img = img.crop((left, top, left + resolution[0], top + resolution[1]))                      

            views.append({
                'img': img,  # Pass the processed PIL image
                'label': f"{scene_name}/{osp.basename(img_path)}", 'instance': f"{scene_idx}_{i}",
                'is_metric': self.is_metric, 'is_video': ordered_video,
                'single_view': False, 'reset': False,
            })

        # Handle cases where not enough views could be loaded
        if len(views) != num_views:
            print(f"Warning: Could not load the requested {num_views} views for index {idx}. Found {len(views)}.")
            # Fallback: try to load the next sample to avoid crashing the DataLoader
            return self._get_views((idx + 1) % len(self), resolution, rng, num_views)
        
        return views