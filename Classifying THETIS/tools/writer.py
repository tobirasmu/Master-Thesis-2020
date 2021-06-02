import cv2
import numpy as np
import torch as tc
import os
import tools.data_loader as dl


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
        x = x.unsqueeze(0)
        ch, num_frames, height, width = x.shape
    else:
        ch, num_frames, height, width = x.shape
    writer = cv2.VideoWriter(out_directory + name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), dl.FRAME_RATE,
                             (width, height))
    for i in range(num_frames):
        frame = np.moveaxis(x[:, i, :, :].type(dtype=tc.uint8).numpy(), 0, -1)
        if ch == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()