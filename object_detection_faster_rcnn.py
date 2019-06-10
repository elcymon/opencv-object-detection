# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import time
import os
import ntpath



parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
parser.add_argument('--network',help='Name of network.')
parser.add_argument('--confThreshold',help='Confidence threshold.')
parser.add_argument('--nmsThreshold',help='Non-maximum suppression threshold')
parser.add_argument('--imgSize', help='Dimension of image (single number for height and width')
# parser.add_argument('--graph', help='inference graph of the network exported using tf_text_graph...')

args = parser.parse_args()
        
# Load names of classes
# classesFile = "litter/litter.names"
# classes = None
# with open(classesFile, 'rt') as f:
#     classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
if args.network:
    modelConfiguration = "{}.pb".format(args.network)#.split('_')[0]) # network is of form name_iterations
    graph = "{}.pbtxt".format(args.network)

    # modelWeights = "{}.pb".format(args.network)
else:
    print('use --network argument to parse network name')
    sys.exit(1)
# if not args.graph:
#     print('use --graph argument to parse exported pbtxt file using tf_graph_... script')
#     sys.exit(1)

# Initialize the parameters
if args.confThreshold:
    confThreshold = float(args.confThreshold)  #Confidence threshold
else:
    print('use --confThreshold to parse confidence threshold')
    sys.exit(1)
if args.nmsThreshold:
    nmsThreshold = float(args.nmsThreshold)   #Non-maximum suppression threshold
else:
    print('use --nmsThreshold to parse non-maximum suppression threshold')
    sys.exit(1)
if args.imgSize:
    inpWidth = int(args.imgSize)       #Width of network's input image
    inpHeight = int(args.imgSize)      #Height of network's input image
else:
    print('use --imgSize to parse image dimension as integer value')
    sys.exit(1)

outputFile = 'faster_rcnn_py.avi'


# Load a model imported from Tensorflow
# tensorflowNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'labelmap.pbtxt')
net = cv.dnn.readNet(modelConfiguration, graph)

# print(modelConfiguration)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    print(len(layersNames),(net.getUnconnectedOutLayers()))
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def saveDetections(filePath,frame,detections):
    '''
    Saves the detections of a specific frame in the desired filePath directory
    detections is a list of class names and box lists
    frame is the frame number in video or image name
    '''

    with open(filePath + '/{}.txt'.format(frame), 'w+') as f:
        for i in detections:
            f.write(i + '\n')

def postprocess(img,networkOutput):
    # filter invalid detections i.e. too large boxes
    # filter out multiple detections using NMS

    rows, cols, channels = img.shape
    
    classIds = []
    confidences = []
    boxes = []
    idx = 0
    framesDetections = []

    # Loop on the outputs
    for detection in networkOutput[0,0]:
        # print(detection)
        # break
        score = float(detection[2])
        if score > confThreshold:
            
            left = int(detection[3] * cols)
            top = int(detection[4] * rows)
            right = int(detection[5] * cols)
            bottom = int(detection[6] * rows)
    
            
            boxes.append([left, top, right - left, bottom - top])

            confidences.append(float(score))

            classIds.append(idx)

            idx = idx + 1
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left, top, width, height = box
        right = left + width
        bottom = top + height

        #draw a red rectangle around detected objects
        cv.rectangle(img, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        frameDetection = '{} {} {} {} {}'.format('litter',left,top,right,bottom)
        
        framesDetections.append(frameDetection)
    
    return framesDetections

            


        
# Process inputs

# needed for displaying video in window
# winName = 'Deep learning object detection in OpenCV'
# cv.namedWindow(winName, cv.WINDOW_NORMAL)

if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    vidname = ntpath.basename(args.video)
    outputFile = "{}-{}-th{}-nms{}-iSz{}".format(vidname[:-4],args.network,\
    (args.confThreshold).replace('.','p'),(args.nmsThreshold).replace('.','p'),\
        args.imgSize)

    vidDirName = os.path.dirname(args.video)
    outputFolder = vidDirName + '/' + outputFile
    os.mkdir(outputFolder)

else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFolder + '/' + outputFile + '.avi', cv.VideoWriter_fourcc('M','J','P','G'), round(cap.get(cv.CAP_PROP_FPS)),
     (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

loopCount = 0
nFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
startT = time.time()

        
with open(outputFolder + '/' + outputFile +'.csv','w+') as logData:
    logData.write('Frame,Time,loopDuration,InferenceTime\n')
    while cv.waitKey(1) < 0:
        loopStart = time.time()
        
        # get frame from the video
        hasFrame, img = cap.read()
        
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            # Release device
            cap.release()
            break

        
        
        # Input image
        # img = cv2.imread('img.jpg')
        
        # Use the given image as input, which needs to be blob(s).
        net.setInput(cv.dnn.blobFromImage(img, size=(inpWidth, inpHeight), swapRB=True, crop=False))
        loopCount = loopCount + 1
        try:
            # Runs a forward pass to compute the net output
            networkOutput = net.forward()#getOutputsNames(net))
            
            
            
        except:
            print('ERROR Occured')
            continue
        # print(networkOutput)
        # break
        
        # postprocess is responsible for filtering invalid detections and drawing valid predictions
        frameOutput = postprocess(img,networkOutput)
        
        saveDetections(outputFolder,loopCount,frameOutput)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        infTime = t * 1000.0 / cv.getTickFrequency()
        label = 'Inference time: %.2f ms' % (infTime)
        cv.putText(img, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Write the frame with the detection boxes
        if (args.image):
            cv.imwrite(outputFile, img.astype(np.uint8))
        else:
            vid_writer.write(img.astype(np.uint8))
        
        logInfo = "{},{:.4f},{:.4f},{:.4f}\n".format(loopCount, time.time() - startT, time.time() - loopStart, infTime)
        
        print(logInfo,end='')
        logData.write(logInfo)
        # cv.imshow(winName, frame)
