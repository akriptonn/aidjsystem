import numpy as np
import operator
import tensorflow as tf
import copy

class PostProcessor:
    def __init__(self, label, thresh):
        self.label = label
        self.label.append('DROP')
        self.thresh = thresh
    
    def __transform__(self, y1):
        return np.array([self.label[isi] for isi in y1])

    def __average__(self, inp, outp, cls):
        t_arr = []
        t_arr_inp = []
        t__arr = []
        for isi in range(len(self.label)):
            t__arr.append(0)
        t__arr[outp[0]] += 1 
        t_mem = inp[0]
        t_cnt = 1
        t_cnt_arr = 1
        t_arr.append(t__arr)
        t_arr_inp.append(inp[0])
        while (t_cnt<(len(inp))):
            if (t_mem != inp[t_cnt]):
                t_cnt_arr += 1
                t_mem = inp[t_cnt]
                t__arr = []
                for isi in range(len(self.label)):
                    t__arr.append(0)
                while (len(t_arr)<t_cnt_arr):
                    t_arr.append(t__arr)
                    t_arr_inp.append("")
            t_arr_inp[t_cnt_arr-1] = inp[t_cnt]
            t_arr[t_cnt_arr-1][outp[t_cnt]] += 1
            t_cnt +=1
        
        for idx in range(len(t_arr)):
            if (cls=='mode'):
                index, dump = max(enumerate(t_arr[idx]), key=operator.itemgetter(1)) #get max value with it index
                t_arr[idx] = index
            else:
                raise ValueError("ONLY cls=mode argument that available for now!")
        
        return t_arr_inp, t_arr

    def __thresh_enforce__(self, outp):
        other_class_idx = tf.cast(len(self.label)-1, tf.int64)
        other_class_idx = tf.tile( \
        tf.expand_dims(other_class_idx, 0), \
        [tf.shape(outp)[0]] \
        )
        is_other = tf.math.reduce_max(tf.cast(outp >= self.thresh, tf.int8), axis=1)
        predictions = tf.where( \
                 is_other>0, \
                 tf.argmax(outp, 1), \
                 other_class_idx \
                )
        return predictions


    def process(self, inp, outp, cls='mode'):
        t1, t2 = self.__average__(inp, self.__thresh_enforce__(outp), cls)
        return (t1,t2)

