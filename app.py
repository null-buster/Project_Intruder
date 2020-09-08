import argparse
import cv2
#import matplotlib.pyplot as plt
#import matplotlib.animation as animate
#import time 
from inference import Network


INPUT_STREAM = "test_video.mp4"
CPU_EXTENSION = "/home/rahul/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
#COUNT_OF_TRAFFIC = 0
def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The type of the input file"
    d_desc = "The device name, if not 'CPU'"
    conf_desc = "The minimum probability to filter weak detections"
    col_desc  = "The color of bounding boxes: RED, GREEN, BLUE"

    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       2) The user choosing the color of the bounding boxes
    
    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-ct", help=conf_desc, default=0.5)
    optional.add_argument("-c", help=col_desc, default='GREEN')
    args = parser.parse_args()

    return args

def get_color(color):
    
    if(color == 'GREEN'):
        return (0, 255, 0)
    elif(color == 'RED'):
        return (0, 0, 255)
    else:
        return (255, 0, 0)
        
def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    #global COUNT_OF_TRAFFIC

    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= 0.5:
            #COUNT_OF_TRAFFIC += 1
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), get_color(args.c), 1)
            cv2.putText(frame, 'Intruder Alert',(xmin, ymin),cv2.FONT_HERSHEY_SIMPLEX, 0.4, get_color(args.c), 1, cv2.LINE_AA)
    return frame


def infer_on_video(args):
    ### TODO: Initialize the Inference Engine
    plugin = Network()

    ### TODO: Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()
    
    ### TODO: Handle image, video or webcam
    # Create a flag for single images
    image_flag = False
    # Check if the input is a webcam
    if args.i == 'CAM':
        args.i = 0
    elif args.i.endswith('.jpg') or args.i.endswith('.bmp'):
        image_flag = True


    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Create a video writer for the output video
    #if not image_flag:
        # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
        # on Mac, and `0x00000021` on Linux
        # 100x100 to match desired resizing
        #out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (100,100))
    #else:
        #out = None

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width,height))

    #fig = plt.figure()
    
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Perform inference on the frame
        plugin.async_inference(p_frame)

        ### TODO: Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
            #print(result)
            ### TODO: Update the frame to include detected bounding boxes
            frame = draw_boxes(frame, result, args, width, height)
            # Write out the frame

        ### TODO: Write out the frame, depending on image or video
        if image_flag:
            cv2.imwrite('output_image.jpg', frame)
        else:
            out.write(frame)
            cv2.imshow('frame', frame)
            '''plt.bar(time.localtime(), COUNT_OF_TRAFFIC)
            plt.xlabel('Time')
            plt.ylabel('Count of Traffic')
            #plt.xticks(time.localtime(seconds), COUNT_OF_TRAFFIC, fontsize=4, rotation=30)
            plt.show()'''
            
            
            
        # Break if escape key pressed
        if key_pressed == 27:
            break
        '''else:
            _ = animate.FuncAnimation(fig, infer_on_video )'''

    # Release the out writer, capture, and destroy any OpenCV windows
    if not image_flag:
        out.release()

    #out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
