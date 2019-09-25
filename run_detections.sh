# if (($1)); then
#     cd /mnt
#     echo "HPC"
# else
#     echo "local"
# fi


#python3 object_detection_yolo.py --video=videos/20190111GOPR9027.MP4 --network=yolov3-tiny-litter_10000 --confThreshold=0.1 --nmsThreshold=0.0 --imgSize=416
#python3 object_detection_yolo.py --video=videos/20190111GOPR9027.MP4 --network=yolov3-tiny-litter_10000 --confThreshold=0.1 --nmsThreshold=1.0 --imgSize=416
#python3 object_detection_yolo.py --video=videos/20190111GOPR9027qtr.MP4 --network=yolov3-tiny-litter_10000 --confThreshold=0.1 --nmsThreshold=0.0 --imgSize=416
#python3 object_detection_yolo.py --video=videos/20190111GOPR9027qtr.MP4 --network=yolov3-tiny-litter_10000 --confThreshold=0.1 --nmsThreshold=1.0 --imgSize=416
# export OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES=1
# export OPENCV_OPENCL_DEVICE=NVIDIA:GPU:660
# export OPENCV_OCL4DNN_CONFIG_PATH=~/.cache/opencv/4.1-dev/opencl_cache/NVIDIA_Corporation--GeForce_GTX_660--430_26
# python3 object_detection_yolo.py --video=videos/litter-recording/GOPR9067.MP4 --network=yolov3-litter_10000 --confThreshold=0.0 --nmsThreshold=0.0 --imgSize=608


# loop through all MP4 files in folder and perform forward inference
videosPath=videos/litter-recording
network=$1
confThreshold=$2
imgSize=$3
folders="20190111GOPR9027-hflip.MP4 20190111GOPR9027.MP4 20190111GOPR9029-hflip.MP4 20190111GOPR9028-hflip.MP4 20190111GOPR9028.MP4"
# for d in $(cd $videosPath; ls -r *.MP4); do
for d in ${folders}; do
    echo "$d"
    python3 object_detection_yolo.py --video=$videosPath/$d --network=yolov3-litter_10000 --confThreshold=0.0 --nmsThreshold=0.0 --imgSize=608
    #python3 object_detection_faster_rcnn.py --video=$videosPath/$d --network=mobilenetSSD-10000 --confThreshold=0.5 --nmsThreshold=0.0 --imgSize=124
done
