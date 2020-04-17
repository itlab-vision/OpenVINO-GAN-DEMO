"""
Sample of running for running digits GAN model using OpenVINO
"""
import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
import logging as log
from openvino.inference_engine import IENetwork, IECore


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=str)
    parser.add_argument('-w', '--weights', required=True, type=str)
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    ie = IECore()

    # Create net to generate numbers
    log.info("Load model")
    net = IENetwork(model=args.model, weights=args.weights)
    exec_net = ie.load_network(net, "CPU")

    # Set input and output of our model
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    # Predict for random noise of normal distribution
    log.info("Run model")
    noise = np.random.normal(loc=0, scale=1, size=[100, 100])
    generated_images = exec_net.infer(inputs={input_blob: noise})

    log.info("Show results")
    # Show the result of out generation
    generated_images = generated_images['Tanh']
    generated_images = generated_images.reshape(100, 28, 28)
    figsize = (10, 10)
    dim = (10, 10)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    sys.exit(main())
