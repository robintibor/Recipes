import lasagne
import numpy as np
import theano.tensor as T

class StrideReshapeLayer2D(lasagne.layers.Layer):
    def __init__(self, incoming, stride, invalid_fill_value=0, **kwargs):
        self.stride = stride
        self.invalid_fill_value = invalid_fill_value
        super(StrideReshapeLayer2D, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        return reshape_topo(input, self.input_shape, self.stride,
            invalid_fill_value=self.invalid_fill_value)

    def get_output_shape_for(self, input_shape):
        n_dim_0 = input_shape[0]
        if n_dim_0 is not None:
            n_dim_0 = n_dim_0 * self.stride[0] * self.stride[1]
        n_dim_2 = int(np.ceil(input_shape[2] / float(self.stride[0])))

        n_dim_3 = int(np.ceil(input_shape[3] / float(self.stride[1])))

        return (n_dim_0, input_shape[1], n_dim_2, n_dim_3)
    


def reshape_topo(topo_sym, topo_shape, stride, invalid_fill_value=np.nan):
    n_dim_0 = topo_shape[0]
    if n_dim_0 is None:
        n_dim_0 = topo_sym.shape[0]
    n_dim_1 = topo_shape[1]

    n_dim_2 = int(np.ceil(topo_shape[2] / float(stride[0])))

    n_dim_3 = int(np.ceil(topo_shape[3] / float(stride[1])))

    trial_increase = stride[0] * stride[1]

    out_shape = (n_dim_0 * trial_increase, n_dim_1, n_dim_2, n_dim_3)

    out_topo = T.ones(out_shape, dtype=np.float32) * invalid_fill_value

    for i_stride_0 in xrange(stride[0]):
        this_n_dim_2 = int(np.ceil((topo_shape[2] - i_stride_0) / float(stride[0])))
        for i_stride_1 in xrange(stride[1]):
            this_n_dim_3 = int(np.ceil((topo_shape[3] - i_stride_1) / float(stride[1])))
            i_combined_stride = i_stride_0 * stride[1] + i_stride_1
            out_topo = T.set_subtensor(
                out_topo[i_combined_stride::trial_increase,:,:this_n_dim_2, :this_n_dim_3],
                topo_sym[:,:,i_stride_0::stride[0], i_stride_1::stride[1]])
    return out_topo

class FinalMeanLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(FinalMeanLayer,self).__init__(incoming, **kwargs)

    def get_output_for(self, input, input_var=None, **kwargs):
        if input_var is None:
            input_var = lasagne.layers.get_all_layers(self)[0].input_var
        input_shape = lasagne.layers.get_all_layers(self)[0].shape
        if input_shape[0] is not None:
            n_trials = input_shape[0]
        else:
            n_trials = input_var.shape[0]
            
        reshaped_out = input.reshape((n_trials,-1, self.input_shape[1], 
                          self.input_shape[2], self.input_shape[3]))

        meaned_out = T.mean(reshaped_out, axis=(1,3,4))
        
        return meaned_out
        
    def get_output_shape_for(self, input_shape):
        return [None, input_shape[1]]