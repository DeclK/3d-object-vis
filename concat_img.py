import matplotlib.pyplot as plt
import cv2 as cv
import os 
from pathlib import Path
from tqdm import tqdm

output_path = Path('.') / 'output'
img_path = output_path / 'img'
velo_path = output_path / 'velo'
concat_path = output_path / 'concat'
concat_path.mkdir(parents=True, exist_ok=True)

img_list = os.listdir(img_path)
velo_list = os.listdir(velo_path)
pbar = tqdm(img_list)
for img in pbar:
    camera = cv.imread(str(img_path / img))
    velo = cv.imread(str(velo_path / img))
    factor = velo.shape[1] / camera.shape[1]
    camera = cv.resize(camera, dsize=None, fx=factor, fy=factor)
    velo[:camera.shape[0],:camera.shape[1]] = camera
    cv.imwrite(str(concat_path / img), velo)