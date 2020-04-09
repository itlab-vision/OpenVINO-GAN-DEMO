import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
import logging as log
from openvino.inference_engine import IENetwork, IECore
from PIL import Image



def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=str)
    parser.add_argument('-w', '--weights', required=True, type=str)
    parser.add_argument('-i', '--image', required=True, type=str)
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    ie = IECore()

    image = Image.open(args.image)
    h = image.height
    w = image.width
    image = image.resize((224, 224))
    x = np.array(image).astype('float32')
    x = np.transpose(x, [2, 0, 1])
    log.info("Load model")
    net = IENetwork(model=args.model, weights=args.weights)
    exec_net = ie.load_network(net, "CPU")

    # Set input and output of our model
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))


    log.info("Run model")
    result = exec_net.infer(inputs={input_blob: x})

    
    
    log.info("Show result")
    result = result[out_blob]
    out_image = np.array(result[0])
    out_image = out_image.transpose(1,2,0)
    out_image = np.clip(out_image, 0, 255)
    img = Image.fromarray(out_image.astype(np.uint8))
    img = img.resize((w, h))
    
    img.show()



if __name__ == '__main__':
    sys.exit(main())
