# Artifical Oriented Bounding Box (OBB) dataset

Script that creates an artificial dataset of images and coordinates of OBB compatible with YOLOv8 implementation of Ultralytics.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)

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

