import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
#import scorep

def main():
    num_filters = 1
    kernel_size = (2, 2, 2)
    padding = "valid" # valid -> no padding, same -> zero padding


    model = models.Sequential()
    model.add(layers.Conv3D(num_filters, kernel_size, padding=padding, input_shape=(3, 3, 3, 1), name='myconv', use_bias=False))
    weights = model.get_layer('myconv').get_weights()
    print(weights)
    print(weights[0].shape)
    kernel_weights = np.array([[1,2,3,4],[5,6,7,8]]).reshape((2,2,2,1,1))
    model.get_layer('myconv').set_weights([ kernel_weights ])

    model.summary()
    model.save("testConvolution3D.tf")

    input_data = np.array([[
        [[ 1, 2, 3], [ 4, 5, 6], [ 7, 8, 9]],
        [[10,11,12], [13,14,15], [16,17,18]],
        [[19,20,21], [22,23,24], [25,26,27]]
    ]])

    #tf.profiler.experimental.start('tf-test-log')
    #with tf.profiler.experimental.Profile('tf-log-dir'):
    #with scorep.instrumenter.enable():
    output_data = model(input_data)
    #tf.profiler.experimental.stop()

    print(output_data)
    print(model.get_weights())

    #runOptions = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    #runConfig = tf.compat.v1.ConfigProto(run_options=runOptions)
    #runConfigSer = [int(i) for i in runConfig.SerializeToString()]
    #print(runConfigSer)


if __name__ == "__main__":
    print(tf.__version__)
    main()  

# 3D Conv example
#1) 1*1 + 2*2 + 3*4 + 4*5 + 5*10 + 6*11 + 7*13 + 8*14 
#2) 1*2 + 2*3 + 3*5 + 4*6 + 5*11 + 6*12 + 7*14 + 8*15 
#3) 1*4 + 2*5 + 3*7 + 4*8 + 5*13 + 6*14 + 7*16 + 8*17 
#4) 1*5 + 2*6 + 3*8 + 4*9 + 5*14 + 6*15 + 7*17 + 8*18 
#5) 1*10 + 2*11 + 3*13 + 4*14 + 5*19 + 6*20 + 7*22 + 8*23 
#6) 1*11 + 2*12 + 3*14 + 4*15 + 5*20 + 6*21 + 7*23 + 8*24 
#7) 1*13 + 2*14 + 3*16 + 4*17 + 5*22 + 6*23 + 7*25 + 8*26 
#8) 1*14 + 2*15 + 3*17 + 4*18 + 5*23 + 6*24 + 7*26 + 8*27 