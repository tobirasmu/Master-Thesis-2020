import cv2
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
from video_functions import showFrame
from load_THETIS import loadVideo

position = (0, 0, 100)
frame_pos = (0, 0, 50)
pic_pos = (0, 0, 0)


def rot_mat(theta_x, theta_y, backwards=False):
    theta_x = np.deg2rad(theta_x)
    theta_y = np.deg2rad(theta_y)
    rot_x = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    if backwards:
        return np.matmul(rot_x, rot_y)
    else:
        return np.matmul(rot_y, rot_x)


def p_nvec(theta_x, theta_y):
    return np.matmul(rot_mat(theta_x, theta_y), np.array([0, 0, 1]))


def get_pos_in_pic(pos, r_mat, n_vec):
    lambda_ = - n_vec[2] * position[2] / (n_vec[0] * pos[0] + n_vec[1] * pos[1] - n_vec[2] * frame_pos[2])
    point = np.array([lambda_ * pos[0], lambda_ * pos[1], position[2] - lambda_ * frame_pos[2]])
    return np.matmul(r_mat, point)


def pic_perspective(picture, theta_x, theta_y):
    ro_mat = rot_mat(-theta_x, -theta_y, backwards=True)
    np_vec = p_nvec(theta_x, theta_y)
    ch, height, width = picture.shape
    R = np.arange(-7, 7, 0.01)
    N = R.shape[0]
    out_pic = tc.ones((ch, N, N)) * 255
    for i in range(N):
        for j in range(N):
            this_pos = get_pos_in_pic([R[i], R[j]], ro_mat, np_vec)
            if 10 > this_pos[0] > -10:
                if 10 > this_pos[1] > -10:
                    out_pic[:, i, j] = picture[:, int((this_pos[0] + 10) * height / 20),
                                       int((this_pos[1] + 10) * width / 20)]
    return out_pic


video = loadVideo("/Users/Tobias/Desktop/Data/VIDEO_RGB/forehand_slice/p10_fslice_s3.avi", b_w=False, normalize=False)
RGB = video[:, 36, :, :]
# The full RGB part:
cv2.imwrite("/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Pics/04_Data/full.jpg", cv2.cvtColor(np.moveaxis(np.array(pic_perspective(RGB, 45, 30)), 0, -1), cv2.COLOR_RGB2BGR))

# The red part
pic = tc.zeros(RGB.shape)
pic[0, :, :] = RGB[0, :, :]
out_im = cv2.cvtColor(np.moveaxis(np.array(pic_perspective(pic, 45, 30)), 0, -1), cv2.COLOR_RGB2BGR)
save_dir = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Pics/04_Data/red_part.jpg"
cv2.imwrite(save_dir, out_im)

# The green part
pic = tc.zeros(RGB.shape)
pic[1, :, :] = RGB[1, :, :]
out_im = cv2.cvtColor(np.moveaxis(np.array(pic_perspective(pic, 45, 30)), 0, -1), cv2.COLOR_RGB2BGR)
save_dir = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Pics/04_Data/green_part.jpg"
cv2.imwrite(save_dir, out_im)

# The blue part
pic = tc.zeros(RGB.shape)
pic[2, :, :] = RGB[2, :, :]
out_im = cv2.cvtColor(np.moveaxis(np.array(pic_perspective(pic, 45, 30)), 0, -1), cv2.COLOR_RGB2BGR)
save_dir = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Pics/04_Data/blue_part.jpg"
cv2.imwrite(save_dir, out_im)

## The depth frame
video = loadVideo("/Users/Tobias/Desktop/Data/VIDEO_Depth/forehand_slice/p10_fslice_depth_s3.avi", normalize=False)
depth = video[0, 36, :, :]
out_im = cv2.cvtColor(np.moveaxis(np.array(pic_perspective(depth.unsqueeze(0), 45, 30)), 0, -1), cv2.COLOR_RGB2BGR)
save_dir = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Pics/04_Data/depth_part.jpg"
cv2.imwrite(save_dir, out_im)
