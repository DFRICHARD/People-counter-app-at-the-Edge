"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


# MQTT server environment variables
import os
import sys
import time
import socket
import json
import cv2
import logging as log
import paho.mqtt.client as mqtt
from argparse import ArgumentParser
from inference import Network
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def draw_boxes(frame, result):
    '''
        Draw bounding boxes around object when its probability
        is more than the specified one
        @param: frame The original input frame where to draw the boxes.
        @param: result The inferencing result.

    '''

    detected = 0
    for obj in result[0][0]:
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * width)
            ymin = int(obj[4] * height)
            xmax = int(obj[5] * width)
            ymax = int(obj[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (174, 32, 141), 2)
            detected += 1

    # Return original frame with box on it and
    # the number of object (people) on the current frame
    return frame, detected


def connect_mqtt():
    ### Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """

    input_type = args.input
    single_image_mode = False
    request_id = 0
    time_count = 0
    pre_time = 0
    counter = 0
    last_count = 0
    current_count = 0
    total_count = 0
    duration = 0
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    global prob_threshold
    prob_threshold = args.prob_threshold

    ### Load the model through `infer_network` ###
    n, c, h, w = infer_network.load_model(
        args.model, request_id, args.device, args.cpu_extension)[1]

    ### Handle the input stream ###
    if input_type == 'CAM':
        input_stream = 0

    # Check for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_stream = input_type

    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    # Get and open video capture
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)

    # Grab the shape of the input
    global width, height
    width = int(cap.get(3))
    height = int(cap.get(4))

    ### Loop until stream is over ###
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### Read from the video capture ###
        image = cv2.resize(frame, (w, h))

        ### Pre-process the image as needed ###
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))

        ### Start asynchronous inference for specified request ###
        inf_start = time.time()

        infer_network.exec_net(image, request_id)

        ### Wait for the result ###
        if infer_network.wait(request_id) == 0:
            det_time = time.time() - inf_start

            ### Get the results of the inference request ###
            result = infer_network.get_output(request_id)
            inference_time_message = "Inference time: {:.3f}ms".format(det_time*1000)
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (174, 32, 141)
            cv2.putText(frame, inference_time_message,(20, 20), font, 0.6, color, 1)

            ### Extract any desired stats from the results ###
            frame, detected = draw_boxes(frame, result)

            ### Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            if detected != counter:
                last_count = counter
                counter = detected
                if time_count >= 3:
                    pre_time = time_count
                    time_count = 0
                else:
                    time_count = pre_time + time_count
                    pre_time = 0
            else:
                time_count += 1
                if time_count >= 10:
                    current_count = counter
                    if time_count == 20 and current_count > last_count:
                        total_count += current_count - last_count
                        client.publish("person", json.dumps({"total_counts": total_count}))
                    elif time_count == 20 and current_count < last_count:
                        duration = int(pre_time)
                        client.publish('person/duration', json.dumps({'duration': duration}))
                
            client.publish("person", json.dumps({"count": current_count}))


            # Break if escape key pressed
            if key_pressed == 27:
               break

        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    # Disconnect from MQTT
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
