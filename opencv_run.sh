hpc=$1

if ((hpc)); then
    module load opencv
    module load singularity
    singularity exec --bind /nobackup/scsoo/darknet/:/mnt opencv.simg cd /mnt;./run_detections.sh $hpc
else
    singularity exec --bind $PWD:$PWD opencv.simg ./run_detections.sh $hpc
fi

