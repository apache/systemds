# Time Series Alignment Library

## Data Preparation
Out of the box we support two different types of data loaders. One for video data and one for sensor data in form of a
csv file where the first column contains the timestamp. An example .csv file can be found in the data directory. 
In order to use the alignment library two datasets in time-series representation are required. 
In addition to the existing data loaders one can extend the `DataLoader` class to create a time series from other 
modalities. 

## Alignment
We provide different alignment methods to align two datasets that contain either a static or a dynamic time lag. 
See `alignment_test.py` for specific usage of the alignment methods.


## Limitations
Be aware that aligning two video datasets can yield long runtime depending 
on the length and framerate of the video as well as the frame size.