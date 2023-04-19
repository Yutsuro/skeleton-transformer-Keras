from keras import backend as K
from keras.layers import Layer

class SkeletonTransformer(Layer):
    def __init__(self, timesteps, kpts_dim, output_dim, **kwargs):
        '''
        Parameters:
            timesteps: Timesteps of input time-series data (equal to number of frames, mentioned as 'T' in the paper)
            kpts_dim: Dimentions of keypoints (usually 2 (x, y) or 3 (x, y, z))
            output_dim: Dimentions of output (mentioned as 'M' in the paper)
        Input:
            x: 3-dimentional tensor of shape (batchsize, timesteps, kpts_dim*N) where N is number of joints
        '''
        super(SkeletonTransformer, self).__init__(**kwargs)
        self.timesteps = timesteps
        self.kpts_dim = kpts_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        if (input_shape[2] % self.kpts_dim) != 0:
            raise ValueError("The dimentions of keypoints must be a divisor of input_shape[2].")
        self.W = self.add_weight(name='SkeletonTransformer',
                                      shape=(int(input_shape[2]/self.kpts_dim), self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(SkeletonTransformer, self).build(input_shape)

    def call(self, x):
        x = K.reshape(x, (-1, self.timesteps, int(x.shape[2]/self.kpts_dim), self.kpts_dim))
        x = K.permute_dimensions(x, (0, 1, 3, 2))
        x = K.dot(x, self.W)
        x = K.permute_dimensions(x, (0, 1, 3, 2))
        x = K.reshape(x, (-1, self.timesteps, self.output_dim*self.kpts_dim))
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.timesteps, self.output_dim*self.kpts_dim)