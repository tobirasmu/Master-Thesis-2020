import numpy as np
import torch as tc
import cv2
import csv
import os

FRAME_RATE = 18


# %% Function for loading a single video
def loadVideo(filename, middle=None, length=None, b_w=True):
    """
    Loads a video using the full path and returns a 4D tensor (channels, frames, height, width).
    INPUT:
        filename - the full path of the video
        middle - the time at the point of the middle of the shot in the video
        length - the desired length to take out in seconds. Half this time will be included on each side of the middle
        b_w    - true if a black/white video is wanted (only one channel)
    Output:
        A tensor of dimension (channels, frames, height, width) where the number of frames = length * 18 + 1
    """
    cap = cv2.VideoCapture(filename)
    if middle is None:
        numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        firstFrame = 0
        lastFrame = numFrames
    else:
        numFrames = int(length * FRAME_RATE + 1)
        firstFrame = int((middle - length / 2) * FRAME_RATE)
        lastFrame = int(firstFrame + length * FRAME_RATE)
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
            else:
                frame = np.moveaxis(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), -1, 0)
            frames[:, framesLoaded, :, :] = tc.tensor(frame)
            framesLoaded += 1
    cap.release()
    return frames


# %% Function for loading an entire directory
def loadShotType(shot_type, directory, input_file=None, length=None, ignore_inds=None):
    """
    Loads all the videos of a directory and makes them into a big tensor. If the inputfile is given, the output will be
    a big tensor of shape (numVideos, channels, numFrames, height, width), otherwise the output will be a list of 4D
    tensors with different number of Frames.
    Inputfile contains the middle of the shot (time), dontLoadsInds are the videos that will not be loaded, due to
    potential problems.
    Shot types :
     - 0  Forehand flat
     - 1  Backhand
    Output:
        Tensor of dimension (numVideos, channels, numFrames, height, width) where the 3 first channels correspond to the
        RGB video, while the last channel is the black/white depth video.
    """
    shot_types = {
        0: "forehand_flat/",
        1: "backhand/"
    }
    directory_rgb = directory + "VIDEO_RGB/" + shot_types.get(shot_type)
    directory_dep = directory + "VIDEO_Depth/" + shot_types.get(shot_type)
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
            thisRGB = loadVideo(directory_rgb + filenames_rgb[i], b_w=False)
            thisDep = loadVideo(directory_dep + filenames_dep[i], b_w=True)
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
        thisRGB = loadVideo(directory_rgb + filenames_rgb[0], float(files[0][1]), length=length, b_w=False)
        thisDep = loadVideo(directory_dep + filenames_dep[0], float(files[0][1]), length=length, b_w=True)
        thisVideo = tc.cat((thisRGB, thisDep), 0)
        all_videos = tc.empty((numVideos, *thisVideo.shape))
        all_videos[0] = thisVideo
        for i in range(1, numVideos):
            middle = float(files[i][1])
            thisRGB = loadVideo(directory_rgb + filenames_rgb[i], middle=middle, length=length, b_w=False)
            thisDep = loadVideo(directory_dep + filenames_dep[i], middle=middle, length=length, b_w=True)
            all_videos[i] = tc.cat((thisRGB, thisDep), 0)
        return all_videos


# %% Writes all names of directory to file
def writeNames2file(directory, out_directory=None):
    """
    Write all the file names of a directory to a file in the out_directory.
    """
    if out_directory is None:
        out_directory = directory
    out_name = out_directory + str.split(directory, '/')[-2] + '_filenames.csv'
    file = open(out_name, 'w')
    writer = csv.writer(file)
    files = sorted(os.listdir(directory))
    for filename in files:
        writer.writerow([filename])
    file.close()


# %% Writes a tensor to a video
def writeTensor2video(x, name, out_directory=None):
    """
    Writes a tensor of shape (ch, num_frames, height, width) or (num_frames, height, width) (for B/W) to a video at the
    given out_directory. If no directory is given, it will just be placed in the current directory with the name.
    """
    if x.type() != 'torch.ByteTensor':
        x = x.type(dtype=tc.uint8)
    if out_directory is None:
        out_directory = os.getcwd() + '/'
    name = name + '.avi'
    if len(x.shape) == 3:
        num_frames, height, width = x.shape
        ch = 1
    else:
        ch, num_frames, height, width = x.shape
    writer = cv2.VideoWriter(out_directory + name, cv2.VideoWriter_fourcc('M','J','P','G'), FRAME_RATE, (width,height))
    for i in range(num_frames):
        frame = np.moveaxis(x[:, i, :, :].type(dtype=tc.uint8).numpy(), 0, -1)
        if ch == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()


# %% Showing a frame
def showFrame(x, title=None):
    """
    Takes in a tensor of shape (ch, height, width) or (height, width) (for B/W) and plots the image using
    matplotlib.pyplot
    """
    if x.type() != "torch.ByteTensor":
        x = x.type(dtype=tc.uint8)
    if len(x.shape) == 2:
        plt.imshow(x.numpy(), cmap="gray")
    else:
        plt.imshow(np.moveaxis(x.numpy(), 0, -1))
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()
