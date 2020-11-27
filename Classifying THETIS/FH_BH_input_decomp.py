# %% Trying to decompose the training tensor via Tucker
"""
Rank estimation does not work on a tensor that big since the required memory is 538 TiB (tebibyte), hence the ranks will
be chosen intuitively (obs, frame, height, width). We are interested in the temporal information hence, the frame dimension
will be given full rank (not decomposed). Since the frames are rather simple (BW depth), the spatial dimensions will not
be given full rank
"""
modes = [0, 2, 3]
ranks = [10, 120, 160]
core, [A, C, D] = partial_tucker(X[:nTrain, 0, :, :, :], modes=modes, ranks=ranks)

# Takes a significant amount of time
# %% The approximation
wh = 50
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(X[wh, 0, 20, :, :], cmap='gray')
approximation = multi_mode_dot(core, [A[wh], C, D], modes=[0, 2, 3])
plt.subplot(1, 2, 2)
plt.imshow(approximation[20], cmap='gray')
plt.show()