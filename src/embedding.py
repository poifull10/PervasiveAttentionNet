from chainer import Variable
from chainer import Chain
from chainer.links import EmbedID
import chainer.functions as F
import numpy as np


class Embedding(Chain):
    def __init__(self, vocab_src_size:int, vocab_dst_size:int, bad_word:int, ds:int, dt:int):
        super().__init__()
        with self.init_scope():
            self.emb_src = EmbedID(vocab_src_size, ds, ignore_label=None) # TODO: impl of ignore_label
            self.emb_dst = EmbedID(vocab_dst_size, dt, ignore_label=None) # TODO: .

    def __call__(self, src_seq, dst_seq):
        """

            [[9, 20, 18, ..., ],
             [0, 2, 101, ..., ],
             ...
                               ]
            などIDのリストがbatch分くると想定
            ただしnp.arrayで、np.int32.

        :param src_seq:
        :param dst_seq:  先頭に必ずBOSを付与
        :return:
            出力は(ch, height, width)
        """
        batch_input_image = None

        for src, dst in zip(src_seq, dst_seq):
            # train用
            embedded_src_tokens = self.emb_src(src)
            embedded_dst_tokens = self.emb_dst(dst)

            input_src_array = F.reshape(embedded_src_tokens, (-1, len(src), 1))
            input_dst_array = F.reshape(embedded_dst_tokens, (-1, 1, len(dst)))

            input_src_array = F.repeat(input_src_array, len(dst), axis=2)
            input_dst_array = F.repeat(input_dst_array, len(src), axis=1)

            concat = F.reshape(F.concat([input_src_array, input_dst_array], axis=0), (1, -1, len(src), len(dst)))

            if not batch_input_image:
                batch_input_image = concat
            else:
                batch_input_image = F.concat([batch_input_image, concat], axis=0)

        return batch_input_image
