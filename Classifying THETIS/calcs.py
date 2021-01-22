

def MNIST_input_numbers(rank, hidden):
    delta = 28 * 28
    parms = rank * delta + rank * (hidden + 1) + hidden * 11
    FLOPs = rank * (2 * delta - 1) + hidden * (2 * rank - 1) + 10 * (2 * hidden - 1)
    return parms, FLOPs


def THETIS_input_numbers(rank, hidden):
    delta = 4 * 28 * 120 * 160
    parms = rank * delta + rank * (hidden + 1) + hidden * 3
    FLOPs = rank * (2 * delta - 1) + hidden * (2 * rank - 1) + 10 * (2 * hidden - 1)
    return parms, FLOPs


MNIST_input_numbers(10, 10)
THETIS_input_numbers(200, 100)