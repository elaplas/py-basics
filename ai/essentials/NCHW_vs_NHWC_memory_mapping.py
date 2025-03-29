import torch



def NCHW21D(shape: list, n_i, ch_i, h_i, w_i):

    ## The memory chunks/strides for first-channel format: (n_ixCxHxW, ch_ixHxW, h_ixW, w_ix1)
    B = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    i = n_i*(C*H*W) + ch_i*(H*W) + h_i*W + w_i
    return i

def NHWC21D(shape:list, n_i, ch_i, h_i, w_i):

    ## The memory chunk/strides for last-channel formant: (n_ixHxWxC, h_ixWxC, w_ixC, ch_ix1)
    B = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    i = n_i*(H*W*C) + h_i*(W*C) + w_i*C + ch_i
    return i

def oneD2NCHW(shape:list, idx) -> tuple:
    B = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    ## "idx" contains the following "n_i*(C*H*W) + ch_i*(H*W) + h_i*W + w_i", so if we perform
    ## devision and modulo consecutively, we can retrieve all 4D information
    n_i = idx // (H*W*C)  ## ith batch is extracted
    idx = idx%(C*H*W)     ## idx contains now only "ch_i*(H*W) + h_i*W + w_i"
    ch_i = idx// (H*W)    ## ith channel is extracted
    idx = idx%(H*W)       ## idx contains now only "h_i*W + w_i"
    h_i = idx//W          ## ith height/row is extracted
    w_i = idx%W           ## ith column is extracted

    return n_i, ch_i, h_i, w_i

def oneD2NHWC(shape:list, idx) -> tuple:
    B = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    ## "idx" contains the following "n_i*(C*H*W) + h_i*(W*C) + w_i*C + ch_i", so if we perform
    ## devision and modulo consecutively, we can retrieve all 4D information

    n_i = idx // (H*W*C)  ## ith batch is extracted
    idx = idx%(C*H*W)     ## idx contains now only "ch_i*(W*C) + w_i*C + ch_i"
    h_i = idx//(W*C)      ## ith height/row is extracted
    idx = idx%(W*C)       ## idx contains now only "w_i*C + ch_i"
    w_i = idx//C          ## ith height/row is extracted
    ch_i = idx%C           ## ith column is extracted

    return n_i, ch_i, h_i, w_i


def MapNCHW2Memory(A: torch.Tensor) -> list:
    B = A.shape[0]
    C = A.shape[1]
    H = A.shape[2]
    W = A.shape[3]

    memory = [0.0 for _ in range(B*C*H*W)]

    for b_i in range(B):
        for ch_i in range(C):
            for h_i in range(H):
                for w_i in range(W):
                    i = NCHW21D(A.shape, b_i, ch_i, h_i, w_i)
                    memory[i] = float(A[b_i, ch_i, h_i, w_i])

    return memory


def MapNHWC2Memory(A: torch.Tensor) -> list:
    B = A.shape[0]
    C = A.shape[1]
    H = A.shape[2]
    W = A.shape[3]

    memory = [0.0 for _ in range(B*C*H*W)]

    for b_i in range(B):
        for ch_i in range(C):
            for h_i in range(H):
                for w_i in range(W):
                    i = NHWC21D(A.shape, b_i, ch_i, h_i, w_i)
                    memory[i] = float(A[b_i, ch_i, h_i, w_i])

    return memory


A = torch.arange(48).reshape(2, 4, 2, 3)

print("Tensor shape: ") 
print(A.shape)
print("Tensor data: ") 
print(A)

first_channel_memory = MapNCHW2Memory(A)
last_channel_memory = MapNHWC2Memory(A)

print("first channel format:") 
print(first_channel_memory)
print("..................................")
print("last channel format:") 
print(last_channel_memory)

indeces = oneD2NCHW(A.shape, 15)
assert A[indeces[0], indeces[1], indeces[2], indeces[3]] == first_channel_memory[15]
print("From tensor:" , A[indeces[0], indeces[1], indeces[2], indeces[3]])
print("From firt-channel format memory: ", first_channel_memory[15])

indeces = oneD2NHWC(A.shape, 15)
assert A[indeces[0], indeces[1], indeces[2], indeces[3]] == last_channel_memory[15]
print("From tensor:" , A[indeces[0], indeces[1], indeces[2], indeces[3]])
print("From last-channel format memory: ", last_channel_memory[15])