import torch as tc
from video_networks import Net, compressNet
from video_functions import numFLOPsPerPush, time_conv, time_lin, get_variable
from timeit import repeat
from torch.autograd import Variable

HPC = True

NUM_OBS = 1
SAMPLE_SIZE = 1000
BURN_IN = SAMPLE_SIZE // 10
test = get_variable(Variable(tc.rand((NUM_OBS, 4, 28, 120, 160))))

# %% Full timing of the network including layer-wise

net = Net(4, 28, 120, 160)
if tc.cuda.is_available():
    net = net.cuda()
    print("Using CUDA")
print("Based on {} samples", SAMPLE_SIZE)

fullTime = tc.tensor(repeat("net(test)", globals=locals(), number=1, repeat=(SAMPLE_SIZE + BURN_IN))[BURN_IN:])

# Layer-wise :
layer_time_m, layer_time_s = tc.zeros(5), tc.zeros(5)
layer_time_m[0], layer_time_s[0], _ = time_conv(NUM_OBS, (28, 120, 160), 4, 6, (5, 11, 11), padding=(2, 5, 5),
                                                sample_size=SAMPLE_SIZE)
layer_time_m[1], layer_time_s[1], _ = time_conv(NUM_OBS, (14, 30, 40), 6, 16, (5, 11, 11), padding=(0, 0, 0),
                                                sample_size=SAMPLE_SIZE)
layer_time_m[2], layer_time_s[2], _ = time_lin(NUM_OBS, 2800, 128, sample_size=SAMPLE_SIZE)
layer_time_m[3], layer_time_s[3], _ = time_lin(NUM_OBS, 128, 84, sample_size=SAMPLE_SIZE)
layer_time_m[4], layer_time_s[4], _ = time_lin(NUM_OBS, 84, 2, sample_size=SAMPLE_SIZE)
print("Full time was: {:.3} seconds, while sum of layers was {:.3} seconds".format(tc.mean(fullTime),
                                                                                   tc.sum(layer_time_m)))

# %% Timing of the decomposed network
if tc.cuda.is_available():
    net = net.cpu()
# Loading the parameters of the pretrained network (needs to be after converting the network back to cpu)
if HPC:
    net.load_state_dict(tc.load("/zhome/2a/c/108156/Master-Thesis-2020/Results_hpc/trained_network_92.pt"))
else:
    net.load_state_dict(
        tc.load("/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Results_hpc/trained_network_92.pt"))

netDec = compressNet(net)
if tc.cuda.is_available():
    netDec = netDec.cuda()
    net = net.cuda()

fullTime_dec = tc.tensor(repeat("netDec(test)", globals=locals(), number=1, repeat=(SAMPLE_SIZE + BURN_IN))[BURN_IN:])

# Ranks of the decomposition
r1_c1 = netDec.c1[0].out_channels
r2_c1 = netDec.c1[1].out_channels
r1_c2 = netDec.c2[0].out_channels
r2_c2 = netDec.c2[1].out_channels
r1_l1 = netDec.l1[0].out_features
r2_l1 = netDec.l1[1].out_features
r_l2 = netDec.l2[0].out_features

# Layer-wise timing
layer_time_dec_m, layer_time_dec_s = tc.zeros(12), tc.zeros(12)
layer_time_dec_m[0], layer_time_dec_s[0], _ = time_conv(NUM_OBS, (28, 120, 160), 4, r1_c1, (1, 1, 1), padding=(0, 0, 0),
                                                        bias=False, sample_size=SAMPLE_SIZE)
layer_time_dec_m[1], layer_time_dec_s[1], _ = time_conv(NUM_OBS, (28, 120, 160), r1_c1, r2_c1, (5, 11, 11),
                                                        padding=(2, 5, 5), bias=False, sample_size=SAMPLE_SIZE)
layer_time_dec_m[2], layer_time_dec_s[2], _ = time_conv(NUM_OBS, (28, 120, 160), r2_c1, 6, (1, 1, 1), padding=(0, 0, 0),
                                                        sample_size=SAMPLE_SIZE)
layer_time_dec_m[3], layer_time_dec_s[3], _ = time_conv(NUM_OBS, (14, 30, 40), 6, r1_c2, (1, 1, 1), padding=(0, 0, 0),
                                                        bias=False, sample_size=SAMPLE_SIZE)
layer_time_dec_m[4], layer_time_dec_s[4], _ = time_conv(NUM_OBS, (14, 30, 40), r1_c2, r2_c2, (5, 11, 11),
                                                        padding=(0, 0, 0), bias=False, sample_size=SAMPLE_SIZE)
layer_time_dec_m[5], layer_time_dec_s[5], _ = time_conv(NUM_OBS, (14, 30, 40), r2_c2, 16, (1, 1, 1), padding=(0, 0, 0),
                                                        sample_size=SAMPLE_SIZE)
layer_time_dec_m[6], layer_time_dec_s[6], _ = time_lin(NUM_OBS, 2800, r1_l1, bias=False, sample_size=SAMPLE_SIZE)
layer_time_dec_m[7], layer_time_dec_s[7], _ = time_lin(NUM_OBS, r1_l1, r2_l1, bias=False, sample_size=SAMPLE_SIZE)
layer_time_dec_m[8], layer_time_dec_s[8], _ = time_lin(NUM_OBS, r2_l1, 128, sample_size=SAMPLE_SIZE)
layer_time_dec_m[9], layer_time_dec_s[9], _ = time_lin(NUM_OBS, 128, r_l2, bias=False, sample_size=SAMPLE_SIZE)
layer_time_dec_m[10], layer_time_dec_s[10], _ = time_lin(NUM_OBS, r_l2, 84, sample_size=SAMPLE_SIZE)
layer_time_dec_m[11], layer_time_dec_s[11], _ = time_lin(NUM_OBS, 84, 2, sample_size=SAMPLE_SIZE)

print("Full time was: {:.3} seconds, while sum of layers was {:.3} seconds".format(tc.mean(fullTime_dec),
                                                                                   tc.sum(layer_time_dec_m)))
layer_time_dec_comp = tc.tensor([tc.sum(layer_time_dec_m[0:3]), tc.sum(layer_time_dec_m[3:6]),
                                 tc.sum(layer_time_dec_m[6:9]), tc.sum(layer_time_dec_m[9:11]), layer_time_dec_m[11]])
# %% Theoretical speed-ups based on the number of FLOPs (floating point operations (+, -, *, %))

input_shape = (28, 120, 160)
FLOPs_orig = numFLOPsPerPush(net, input_shape, paddings=[1], pooling=[1, 2], pool_kernels=[(2, 4, 4), (2, 4, 4)])
FLOPs_dcmp = numFLOPsPerPush(netDec, input_shape, paddings=[2], pooling=[3, 6], pool_kernels=[(2, 4, 4), (2, 4, 4)])
dcmp_layer_wise = tc.tensor([tc.sum(FLOPs_dcmp[0:3]), tc.sum(FLOPs_dcmp[3:6]), tc.sum(FLOPs_dcmp[6:9]),
                             tc.sum(FLOPs_dcmp[9:11]), FLOPs_dcmp[11]])

# Calculating layer-wise speed-ups
theoretical_SP_layer = FLOPs_orig / dcmp_layer_wise
observed_SP_layer = layer_time_m / layer_time_dec_comp

print("\n\n{:-^60s}\n{:-^60s}\n{:-^60s}\n".format('', " Timing the networks ", ''))
print("{: <11}{: ^16}{: ^16}{: ^16}\n{:-^60s}".format("Layer", "Theoretical", "Observed", "Accounts for", ''))

layer_names = ["Conv1", "Conv2", "Lin1", "Lin2", "Lin3"]
for i in range(len(FLOPs_orig)):
    print("{: <11s}{: ^16.4f}{: ^16.4f}{: >7.5f} / {: <7.5f}".format(layer_names[i], theoretical_SP_layer[i],
                                                                     observed_SP_layer[i],
                                                                     FLOPs_orig[i] / tc.sum(FLOPs_orig),
                                                                     layer_time_m[i] / tc.sum(layer_time_m)))
print("{:-^60s}\n{: <11s}{: ^16.4f}{: ^16.4f}".format('', "Total", tc.sum(FLOPs_orig) / tc.sum(FLOPs_dcmp),
                                                      tc.mean(fullTime) / tc.mean(fullTime_dec)))
