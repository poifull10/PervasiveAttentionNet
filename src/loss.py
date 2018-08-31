from chainer import Chain
import chainer.functions as F
import numpy as np
from chainer import Variable

def seq_cross_entropy(p, dst_seq, eps=1e-7):
    """
    :param p: Shape:(batch_size, embedding_size, seq_size)
    :param dst_seq:
    :return:
    """
    mask = Variable(np.ones(dst_seq.shape, dtype=np.float32))
    # Paddingなど除去
    mask.data[dst_seq<0] = 0
    return F.sum( - mask * p * F.log(p + eps) )