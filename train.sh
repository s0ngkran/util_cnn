echo
echo "start train..."
echo
echo "python train.py"

wait_gpu() {
    gb=$1
    gb=$(expr $gb '*' 1024) 
    # 20480 == 20GB
    threshold=$gb
    while true; do
        mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1}' | sort -n | tail -1)
        echo "GPU= $mem MB; wait --> $gb MB|$(date -d "+7 hours") Thailand";
        if (( mem > threshold )); then
            break
        fi
        sleep 5;
    done
}

hello(){
    echo "hello";
}

tr(){
    config=$1
    for i in $(seq $2 $3)
        do
            python train.py ${config}_$i --config $config;
        done
}

tr_(){
    config=$1
    i=$2
    python train.py ${config}_$i --config $config;
}

#  -> static model size| change input sigma
# sigma size          -> change size of sigma on all             | tr_720, tr_512, tr_256, tr_128, tr_64
# curriculum learning -> change on each stage of model big2small | tr_p4, tr_p6, tr_p8

# running
# tr s64 0 2& # bach
# tr s128 0 2& # bach

tr_ s256 0& #mozart

# wait_gpu 10;
# tr_ s256 1;
# wait_gpu 10;
# tr_ s256 2;
# wait_gpu 10;
# tr_ s256 3;
# wait_gpu 10;
# tr_ s256 4;
# wait_gpu 10;
# tr_ s256 5;
# wait_gpu 10;
# tr_ s256 6;
