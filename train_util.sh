
TZ='Asia/Bangkok'; export TZ;
date;

wait_gpu() {
    gb=$1
    gb=$(expr $gb '*' 1024) 
    # 20480 == 20GB
    threshold=$gb
    while true; do
        mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1}' | sort -n | tail -1)
        echo "GPU= $mem MB; wait --> $gb MB|$(date) Thailand";
        if (( mem > threshold )); then
            break
        fi
        sleep 60;
    done
}

tr(){
    config=$1
    i=$2
    args=$3
    name=${config}_$i
    echo "train... $name"
     python train.py $name --config $config $args>log/$name;
     echo "done $name";
}

te(){
    name=$1
    args=$2
    python test.py $name $args|grep acc>>acc;
}

