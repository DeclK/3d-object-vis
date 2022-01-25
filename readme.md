This repository is based on [3D-Detection-Tracking-Viewer](https://github.com/hailanyi/3D-Detection-Tracking-Viewer).

## Usage

### Requirement

1. You need to`pip install vedo` to visualize your prediction. For more details, please refere to  [3D-Detection-Tracking-Viewer](https://github.com/hailanyi/3D-Detection-Tracking-Viewer).
2. You need to `pip install tqdm` to show the progress bar.
3. You need to `pip install opencv-python` to concat 3D & 2D images.

### Data

You can put your data like this, just like KITTI

```
- data_root
	- velodyne
	- image_2
```

### Scripts

Main functions are in `vis.py`. Most of the funcionalities are commented, and some were written in Chinese, but overall it's easy to read. Please check the scripts in `vis.py` after `if __name__ == '__main__':` to see the basic usage

## Demo

I've made some demo videos to show the results:

1. [3D Object Detection Visualization Demo](https://www.bilibili.com/video/BV1h3411t7sc)
2. [3D object Segmentation (Car) Demo](https://www.bilibili.com/video/BV1oT4y1f71D)

