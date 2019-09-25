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
parser.add_argument('--network',help='Name of YOLO network.')
parser.add_argument('--confThreshold',help='Confidence threshold.')
parser.add_argument('--nmsThreshold',help='Non-maximum suppression threshold')
parser.add_argument('--imgSize', help='Dimension of image (single number for height and width')

args = parser.parse_args()
        
# Load names of classes
classesFile = "yolo/litter-cfg/litter.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
if args.network:
    modelConfiguration = "yolo/litter-cfg/{}.cfg".format((args.network).split('_')[0]) # network is of form name_iterations
    modelWeights = "yolo/weights/{}.weights".format(args.network)
else:
    print('use --network argument to parse network name')
    sys.exit(1)

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

outputFile = 'yolo_out_py.avi'


net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def saveDetections(filePath,vidname,frameCount,detections,leftTop_rightBottom,frame):
    '''
    Saves the detections of a specific frame in the desired filePath directory
    detections is a list of class names and box lists
    frameCount is the frame number in video or image name
    '''

    # with open(filePath + '/{}.txt'.format(frameCount), 'w+') as f:
    # for i in detections:
    #     # frameDetection = '%s %s %s %s %s %s' % ['litter',confidences[i],left,top,right,bottom]
    #     # print(i)
    #     frameDetection = '%s %s %s %s %s %s' % i
        
    for seg in leftTop_rightBottom:
        segmentFolder = '%s_%s-%s_%s' % seg
        #visualize the segments
        cv.rectangle(frame, (seg[0],seg[1]), (seg[2],seg[3]), (0, 128, 255), 2)

        with open(filePath + '/' + segmentFolder + '/{}-{:05d}.txt'.format(vidname,frameCount), 'a+') as f:
            for i in detections:
                # frameDetection = '%s %s %s %s %s %s' % ['litter',confidences[i],left,top,right,bottom]
                # print(i)
                frameDetection = '%s %s %s %s %s' % (i[0],i[2],i[3],i[4],i[5]) #'%s %s %s %s %s %s' % i #
                        
                if seg[0] <= (i[2] + i[4]) / 2.0 and seg[1] <= (i[3] + i[5]) / 2.0 and \
                    seg[2] > (i[2] + i[4]) / 2.0 and seg[3] > (i[3] + i[5]) / 2.0:
                # if (seg[0] <= i[2] and seg[1] <= i[3] and seg[2] > i[2] and seg[3] > i[3]) \
                #     or (seg[0] <= i[4] and seg[1] <= i[5] and seg[2] > i[4] and seg[3] > i[5]):
                    f.write(frameDetection + '\n')
                # else:
                #     f.write('')


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        # print(classId,classes)
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
        # label = '%s:%s' % (classes[0], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    # cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    # cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1] 

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    framesDetections = []
    detectorBox = []
    networkCoords = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)

            confidence = scores[classId]
            # classId = 0
            # confidence = scores[classId]
            if confidence > confThreshold:
                shift = 1
                center_x = int(detection[0] * frameWidth * shift)
                center_y = int(detection[1] * frameHeight * shift)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                detectorBox.append([detection[0],detection[1],detection[2],detection[3]])
                # print(detection)
                # print(scores)
                # input('>')
            

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    # print(classIds)
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        right = left + width
        bottom = top + height

        drawPred(classIds[i], confidences[i], left, top, right, bottom)
        networkCoords.append(tuple(['litter',confidences[i]] + detectorBox[i]))

        # frameDetection = '%s %s %s %s %s %s' % ['litter',confidences[i],left,top,right,bottom]
        
        framesDetections.append(('litter',confidences[i],left,top,right,bottom))
    return framesDetections,networkCoords

# Process inputs

# needed for displaying video in window
#winName = 'Deep learning object detection in OpenCV'
#cv.namedWindow(winName, cv.WINDOW_NORMAL)

if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    imgname = ntpath.basename(args.image)
    outputFile = "{}-{}-th{}-nms{}-iSz{}".format(imgname[:-4],args.network,\
    (args.confThreshold).replace('.','p'),(args.nmsThreshold).replace('.','p'),\
        args.imgSize)
    imgDirName = os.path.dirname(args.image)
    outputFolder = imgDirName + '/' + outputFile
    os.makedirs(outputFolder, exist_ok=True)
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
    os.makedirs(outputFolder, exist_ok=True)
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFolder + '/' + outputFile + '.avi',cv.CAP_FFMPEG, cv.VideoWriter_fourcc(*'X264'), round(cap.get(cv.CAP_PROP_FPS)),
     (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)/2),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT)/2)))

loopCount = 0
nFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
startT = time.time()
leftTop_rightBottom = []

with open(outputFolder + '/' + outputFile + '.csv','w+') as logData:
    logData.write('Frame,Time,loopDuration,InferenceTime\n')
    while cv.waitKey(1) < 0:
        loopStart = time.time()
        
        # get frame from the video
        hasFrame, frame = cap.read()
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            # Release device
            cap.release()
            break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        loopCount = loopCount + 1
        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))
        # print(outs)
        # break
        
        #resize frame to desired resolution
        frame = cv.resize(frame,\
                (int(frame.shape[1]/2),int(frame.shape[0]/2)),\
                cv.INTER_LANCZOS4)
        
        
        if len(leftTop_rightBottom) == 0:
            #create a 4 boxes by 3 boxes segment of frame. i.e. 4 columns and 3 rows of boxes.
            fHeight = frame.shape[0]
            fWidth = frame.shape[1]
            yPoints = np.linspace(start=0,stop=fHeight,num=2,dtype=np.int,endpoint=True)
            xPoints = np.linspace(start=0,stop=fWidth,num=2,dtype=np.int,endpoint=True)

            #create segments from frame
            for y in range(len(yPoints) - 1):
                for x in range(len(xPoints) - 1):
                    leftTop_rightBottom.append((xPoints[x], yPoints[y], xPoints[x+1], yPoints[y+1]))
                    segmentFolder = '{}_{}-{}_{}'.format(xPoints[x], yPoints[y], xPoints[x+1], yPoints[y+1])

                    #create directory to store segment detections
                    # os.makedirs(outputFolder + '/' + segmentFolder, exist_ok=True)
                    os.makedirs(outputFolder + '/' + segmentFolder + '/analysis',exist_ok=True)

        
        # Remove the bounding boxes with low confidence
        frameOutput,networkCoords = postprocess(frame, outs)
        
        saveDetections(outputFolder, vidname[:-4], loopCount,frameOutput, leftTop_rightBottom, frame)
        # saveDetections(outputFolder,vidname[:-4] + '-ntwk', loopCount,networkCoords,leftTop_rightBottom,frame)
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        infTime = t * 1000.0 / cv.getTickFrequency()
        label = '%d Inference time: %.2f ms' % (loopCount,infTime)
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Write the frame with the detection boxes
        if (args.image):
            cv.imwrite(outputFolder + '/' + outputFile + '.jpg', frame.astype(np.uint8))
        else:
            vid_writer.write(frame.astype(np.uint8))
            # cv.imwrite(outputFolder + '/' + outputFile +'-' + str(loopCount) + '.jpg', frame.astype(np.uint8))
        
        logInfo = "{},{:.4f},{:.4f},{:.4f}\n".format(loopCount, time.time() - startT, time.time() - loopStart, infTime)
        
        print(logInfo.replace(str(loopCount) + ',','{}/{}'.format(loopCount,nFrames)),end='')
        logData.write(logInfo)
        # if loopCount >= 100:
        #     break
#        cv.imshow(winName, frame)
#        cv.waitKey()
cap.release()
