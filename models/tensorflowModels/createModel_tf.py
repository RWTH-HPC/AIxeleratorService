import tensorflow as tf
from typing import List
import argparse


def FlexMLP(n_inp: int, n_out: int, n_hidden_neurons: List[int] = [32, 32], activation_fn = 'relu', dtype='float64', name=None):
    m = tf.keras.Sequential()
    m.add(tf.keras.Input(shape=(n_inp,), dtype=dtype))
    for h in n_hidden_neurons:
        m.add(tf.keras.layers.Dense(h,
                                       activation=activation_fn,
                                       dtype=dtype,
                                       kernel_initializer=tf.keras.initializers.Constant(1.0/n_hidden_neurons[0]),
                                       bias_initializer=tf.keras.initializers.Zeros()))
    m.add(tf.keras.layers.Dense(n_out, dtype=dtype, kernel_initializer=tf.keras.initializers.Constant(1.0/n_hidden_neurons[0]),
                                bias_initializer=tf.keras.initializers.Zeros()))
    return m


def parseArguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('ninp', type=int,
                        help='Number of input neurons')
    parser.add_argument('nout', type=int,
                        help='Number of output neurons')
    parser.add_argument('-n', '--neurons', dest='neurons', nargs='+', default=[32, 32],
                        help='List of neurons in hidden layers, e.g. -n 32 64 32')
    parser.add_argument('--output', '-o', dest='model', required=False, default='Script.tf',
                        help='File name of save tf script, default: Script.tf')

    return parser.parse_args()


def main():
    args = parseArguments()

    n_inp = args.ninp
    n_out = args.nout
    n_neurons = [int(x) for x in args.neurons]

    m = FlexMLP(n_inp, n_out, n_neurons)
    # m.save(args.model)
    m.export(args.model)
   # m.summary()

    #for layer in m.layers: 
    #    print(layer.get_config(), layer.get_weights())

    #test_inputs = [
    #    (0, 0),
    #    (1,-1),
    #    (2,-2),
    #    (3,-3)
    #]

    #inputs = tf.keras.layers.Input(shape=(2,))
    #x = tf.keras.layers.Dense(10, activation='relu', name='myLayer1')(inputs)
    #x = tf.keras.layers.Dense(10, activation='relu', name='myLayer2')(x)
    #outputs = tf.keras.layers.Dense(2, name='myLayer3')(x)

    #model2 = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    #input = tf.constant([[1,1]], dtype='float64')
    #print(model2(input, training=None, mask=None))

    tf.debugging.set_log_device_placement(True)
    physical_devices = tf.config.list_physical_devices()
    for dev in physical_devices:
        print(dev)

    logical_devices = tf.config.list_logical_devices()
    for dev in logical_devices:
        print(dev)

    config = tf.compat.v1.ConfigProto(device_count={"GPU":1})
    print(config)
    ser = config.SerializeToString()
    hexlist = list(map(hex,ser))
    print(hexlist)

    input = tf.constant([[0,0],[1,1],[2,2],[3,3]], dtype='float64')
    y = m(input)
    print(y)

   # for i in range(len(test_inputs)):
    #    res = model(test_inputs[i])
    #    print(f"{test_inputs[i]} --> {res}")

if __name__ == "__main__":
    main()
