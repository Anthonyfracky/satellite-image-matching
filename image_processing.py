import os
import cv2
import warnings
import rasterio
import numpy as np
from tqdm import tqdm
from typing import List, Dict

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def read_large_image_tiles(image_path: str, tile_size: int = 1024, overlap: int = 256) -> List[Dict]:
    with rasterio.open(image_path) as src:
        image = src.read()
        if len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
        else:
            image = image[0]
        image = (image / image.max() * 255).astype(np.uint8)
        tiles = []
        height, width = image.shape[:2]
        total_tiles = ((height - 1) // (tile_size - overlap) + 1) * ((width - 1) // (tile_size - overlap) + 1)
        pbar = tqdm(total=total_tiles, desc=f"Reading tiles ({os.path.basename(image_path)})", unit="tile")
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                tile = image[y:y + tile_size, x:x + tile_size]
                if tile.size > 0 and tile.shape[0] > 10 and tile.shape[1] > 10:
                    tiles.append({'tile': tile, 'x': x, 'y': y})
                pbar.update(1)
        pbar.close()
        print(f"Image {os.path.basename(image_path)} split into {len(tiles)} tiles.\n")
        return tiles


def detect_keypoints_tiles(tiles: List[dict], image_path: str, method: str = 'sift') -> List[dict]:
    detector = cv2.SIFT_create() if method == 'sift' else cv2.ORB_create()
    keypoint_tiles = []
    for tile_data in tqdm(tiles, desc=f"Detecting keypoints ({os.path.basename(image_path)})", unit="tile"):
        gray = cv2.cvtColor(tile_data['tile'], cv2.COLOR_BGR2GRAY) if len(tile_data['tile'].shape) == 3 else tile_data[
            'tile']
        kp, desc = detector.detectAndCompute(gray, None)
        adjusted_kp = [cv2.KeyPoint(k.pt[0] + tile_data['x'], k.pt[1] + tile_data['y'], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp]
        keypoint_tiles.append({'keypoints': adjusted_kp, 'descriptors': desc, 'x': tile_data['x'], 'y': tile_data['y']})
    return keypoint_tiles


def match_global_keypoints(keypoint_tiles1: List[dict], keypoint_tiles2: List[dict], max_matches: int = 1000) -> List:
    print("\nPreparing descriptors for matching...")
    all_desc1 = np.vstack([tile['descriptors'] for tile in keypoint_tiles1 if tile['descriptors'] is not None])
    all_desc2 = np.vstack([tile['descriptors'] for tile in keypoint_tiles2 if tile['descriptors'] is not None])
    print("Matching keypoints...\n")
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(all_desc1, all_desc2)
    filtered_matches = sorted(matches, key=lambda x: x.distance)[:max_matches]
    return filtered_matches


def visualize_matches(image1: np.ndarray, image2: np.ndarray, keypoints1: List[cv2.KeyPoint], keypoints2: List[cv2.KeyPoint], matches: List) -> np.ndarray:
    return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def read_image(image_path: str) -> np.ndarray:
    with rasterio.open(image_path) as src:
        image = src.read()
        image = np.transpose(image, (1, 2, 0))
        return (image / image.max() * 255).astype(np.uint8)


def process_images(image1_path: str, image2_path: str, method: str = 'sift', max_matches: int = 1000) -> Dict:
    tiles1 = read_large_image_tiles(image1_path)
    tiles2 = read_large_image_tiles(image2_path)
    keypoint_tiles1 = detect_keypoints_tiles(tiles1, image1_path, method)
    keypoint_tiles2 = detect_keypoints_tiles(tiles2, image2_path, method)
    matches = match_global_keypoints(keypoint_tiles1, keypoint_tiles2, max_matches)
    return {'matches': matches, 'keypoint_tiles1': keypoint_tiles1, 'keypoint_tiles2': keypoint_tiles2}
