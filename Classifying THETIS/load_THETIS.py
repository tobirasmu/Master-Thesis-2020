# Whether working from the HPC cluster or not
HPC = False

import numpy as np
import cv2
import torch as tc
from torch.nn.functional import interpolate
import os
import csv

LENGTH = 1.5
RESOLUTION = 0.25


# %% Function for loading a single video
def loadVideo(filename, middle=None, nFrames=None, b_w=True, resolution=1., normalize=True):
    """
    Loads a video using the full path and returns a 4D tensor (channels, frames, height, width).
    INPUT:
        filename   - the full path of the video
        middle     - the time at the point of the middle of the shot in the video
        nFrames    - The number of frames to be loaded. Half of this will be included on each side of the middle
        b_w        - True if a black/white video is wanted (only one channel)
        resolution - The desired output resolution
        normalize  - If the output should be normalized
    Output:
        A tensor of dimension (channels, frames, height, width) where the number of frames = length * 18 + 1
    """
    cap = cv2.VideoCapture(filename)
    frameRate = int(cap.get(cv2.CAP_PROP_FPS))
    if middle is None:
        numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        firstFrame = 0
        lastFrame = numFrames
    else:
        numFrames = nFrames
        firstFrame = int(middle * frameRate) - int((numFrames - 1) / 2)
        lastFrame = firstFrame + numFrames - 1
    ch = 1 if b_w else 3
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = tc.empty((ch, numFrames, height, width))
    framesLoaded = 0
    framesRead = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False or framesRead is lastFrame:
            break
        framesRead += 1
        if framesRead >= firstFrame:
            if b_w:
                frame = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0)
                frame = frame / 255 if normalize else frame
            else:
                frame = np.moveaxis(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), -1, 0)
                frame = frame / 255 if normalize else frame
            frames[:, framesLoaded, :, :] = tc.tensor(frame)
            framesLoaded += 1
    cap.release()
    if resolution != 1:
        factor = int(1 // resolution)
        frames = interpolate(frames, size=(height // factor, width // factor))
    return frames


# %% Function for loading an entire directory
def loadShotType(shot_type, in_dir, input_file=None, length=None, ignore_inds=None, resolution=1., normalize=True):
    """
    Loads all the videos of a directory and makes them into a big tensor. If the inputfile is given, the output will be
    a big tensor of shape (numVideos, channels, numFrames, height, width), otherwise the output will be a list of 4D
    tensors with different number of Frames.
    INPUT:
            Shot types     - The type of shot that is to be loaded
                - 0  Forehand flat
                - 1  Backhand
            Inputfile      - File that contains the middle of the shot (s)
            ignore_inds    - The indices of the videos that will not be loaded, due to potential problems.
            length         - The desired length of the output video (s)
            resolution     - The resolution is for getting a lower resolution if needed (< 1).
            in_dir         - The input directory
            out_dir        - The output directory
            normalize      - Whether the data should be normalized or not
    OUTPUT:
        Tensor of dimension (numVideos, channels, numFrames, height, width) where the 3 first channels correspond to the
        RGB video, while the last channel is the black/white depth video.
    """
    shot_types = {
        0: "forehand_flat/",
        1: "backhand/"
    }
    directory_rgb = in_dir + "VIDEO_RGB/" + shot_types.get(shot_type)
    directory_dep = in_dir + "VIDEO_Depth/" + shot_types.get(shot_type)
    filenames_rgb = sorted(os.listdir(directory_rgb))
    filenames_dep = sorted(os.listdir(directory_dep))

    if input_file is None:
        if ignore_inds is not None:
            if len(ignore_inds) > 1:
                ignore_inds = sorted(ignore_inds, reverse=True)
            for ind in ignore_inds:
                filenames_rgb.pop(ind)
                filenames_dep.pop(ind)
        all_videos = []
        for i in range(len(filenames_rgb)):
            thisRGB = loadVideo(directory_rgb + filenames_rgb[i], b_w=False, resolution=resolution, normalize=normalize)
            thisDep = loadVideo(directory_dep + filenames_dep[i], b_w=True, resolution=resolution, normalize=normalize)
            all_videos.append(tc.cat((thisRGB, thisDep), 0))
        return all_videos
    else:
        rFile = open(input_file, "r")
        reader = csv.reader(rFile, delimiter=";")
        files = list(reader)
        if ignore_inds is not None:
            if len(ignore_inds) > 1:
                ignore_inds = sorted(ignore_inds, reverse=True)
            for ind in ignore_inds:
                files.pop(ind)
                filenames_rgb.pop(ind)
                filenames_dep.pop(ind)
        numVideos = len(filenames_rgb)
        # Turning the length into the number of frames due to different frame rates in the videos
        nFrames = int(18 * length + 1)
        thisRGB = loadVideo(directory_rgb + filenames_rgb[0], float(files[0][1]), nFrames=nFrames, b_w=False,
                            resolution=resolution, normalize=normalize)
        thisDep = loadVideo(directory_dep + filenames_dep[0], float(files[0][1]), nFrames=nFrames, b_w=True,
                            resolution=resolution, normalize=normalize)
        thisVideo = tc.cat((thisRGB, thisDep), 0)
        all_videos = tc.empty((numVideos, *thisVideo.shape))
        all_videos[0] = thisVideo
        for i in range(1, numVideos):
            middle = float(files[i][1])
            thisRGB = loadVideo(directory_rgb + filenames_rgb[i], middle=middle, nFrames=nFrames, b_w=False,
                                resolution=resolution)
            thisDep = loadVideo(directory_dep + filenames_dep[i], middle=middle, nFrames=nFrames, b_w=True,
                                resolution=resolution)
            all_videos[i] = tc.cat((thisRGB, thisDep), 0)
        return all_videos


# %% Loading the data and saving as a tensor
def loadTHETIS(shotTypes, input_files, ignore_inds, in_dir, out_dir, length=1.5, resolution=0.25, seed=43):
    """
    Loads all the videos from the given shot types, merges them into a big tensor (randomly) and saves the tensor to a
    file that can later be accessed.
    INPUT:
            shotTypes  - A tuple with the desired shot types
            input_files - A tuple with the full paths to the files containing information about the middle of the shots
            ignore_inds - A tuple with a list of indices to be ignored for each shot type
            in_dir      - Location of the folder containing the Data folder
            out_dir     - The desired output location of the big tensor
            length      - The desired length of the videos
            resolution  - The desired resolution (< 1)
            seed        - Seed for the random permutation of the merged data

    """
    tc.manual_seed(seed)
    this_X = loadShotType(shotTypes[0], in_dir, input_file=input_files[0], length=length, resolution=resolution,
                          ignore_inds=ignore_inds[0])
    this_Y = tc.zeros(this_X.shape[0])
    for i in range(1, len(shotTypes)):
        this = loadShotType(shotTypes[i], in_dir, input_file=input_files[i], length=length, resolution=resolution,
                            ignore_inds=ignore_inds[i])
        this_X = tc.cat((this_X, this), dim=0)
        this_Y = tc.cat((this_Y, tc.ones(this.shape[0]) * i))
    permutation = tc.randperm(this_Y.shape[0])
    this_X = this_X[permutation]
    this_Y = this_Y[permutation]
    tc.save((this_X, this_Y), out_dir)
    print("{:-^60s}".format(" Data is loaded and saved "))
    return this_X, this_Y


# %% Writes all names of directory to file
def writeNames2file(in_dir, out_directory=None):
    """
    Write all the file names of a directory to a file in the out_directory in order to manually annotate the middle of
    each of the shots.
    """
    if out_directory is None:
        out_directory = in_dir
    out_name = out_directory + str.split(in_dir, '/')[-2] + '_filenames.csv'
    file = open(out_name, 'w')
    writer = csv.writer(file)
    files = sorted(os.listdir(in_dir))
    for filename in files:
        writer.writerow([filename])
    file.close()


# %% Actually loading the data and saving it to a file
""" 
FOREHANDS
The file forehand_filenames has information about the middle of the stroke so
that the same amount of features can be extracted from each video.

Seems there are some problems with the depth videos for the following 2 instances:
There are not the same number of frames in the RGB and depth videos respectively.

    p13_foreflat_depth_s2.avi (10) - the one after is identical and works (11)
    p24_foreflat_depth_s1.avi (45)

BACKHANDS
The file backhand_filenames_adapted has information about the middle of the 
stroke.

The very first seems to be wrong. 
    p1_foreflat_depth_s1.avi (0) (RGB is wrong - is actually p50)
"""

if HPC:
    directory = "/zhome/2a/c/108156/Data_MSc/"
    inputForehand = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/forehand_filenames_adapted.csv"
    inputBackhand = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/backhand_filenames_adapted.csv"
    X, Y = loadTHETIS((0, 1), (inputForehand, inputBackhand), ([10, 45], [0]), directory,
                      out_dir=directory + "data.pt", length=LENGTH, resolution=RESOLUTION)
else:
    directory = "/Users/Tobias/Desktop/Data/"
    # Forehands
    inputForehand = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying " \
                    "THETIS/forehand_filenames_adapted.csv "
    # Backhands
    inputBackhand = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying " \
                    "THETIS/backhand_filenames_adapted.csv "
    X, Y = loadTHETIS((0, 1), (inputForehand, inputBackhand), ([10, 45], [0]), directory,
                      out_dir=directory + "data.pt",
                      length=LENGTH, resolution=RESOLUTION)
