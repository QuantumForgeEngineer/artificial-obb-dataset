# Artifical Oriented Bounding Box (OBB) dataset

Script that creates an artificial dataset of images and coordinates of OBB compatible with YOLOv8 implementation of Ultralytics.
![image](https://github.com/QuantumForgeEngineer/artificial-obb-dataset/assets/166127086/c1272a6f-ab71-40e9-9b0d-015c2f39efdc)

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)

## Description

This script will create two folders: images and labels which have three subfolders: train, val and test.
Images are 512x512 in size but can be modified in source code.
The script will choose between 1 and 5 shapes to draw in an image. It will pick randomly from circles and rectangles (see source code) from a custom probability repartition.
The shapes avalaible are circle and rectangle.
All image are grayscale and noise is added at the end of drawing all shapes.

## Installation

```bash
conda create -n aobb python=3.9
conda activate aobb
pip install -r requirements.txt
python main.py
```

## Usage

You can run the main.py after editing the *writing_path* and *num_images* variables.

## Examples
Below examples of images generated

- ![image](https://github.com/QuantumForgeEngineer/artificial-obb-dataset/assets/166127086/52caa3b5-9bce-49da-8559-80721a2800d6)
- ![image](https://github.com/QuantumForgeEngineer/artificial-obb-dataset/assets/166127086/f7b52eca-b0ad-4e4d-949b-b715d480b39e)




