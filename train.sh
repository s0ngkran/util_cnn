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
    args=$3
    python train.py ${config}_$i --config $config $3;
}
trcon(){
    config=$1
    i=$2
    python train.py ${config}_$i --config $config -col -s 70;
}

#  -> static model size| change input sigma
# sigma size          -> change size of sigma on all             | tr_720, tr_512, tr_256, tr_128, tr_64
# curriculum learning -> change on each stage of model big2small | tr_p4, tr_p6, tr_p8

for i in {2..5}
do
    wait_gpu 10;
    tr_ s360 AB${i}_lr1 "-b 5 -lr -1"&
    sleep 300;
done



