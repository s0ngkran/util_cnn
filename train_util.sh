
dateLocal(){
    echo "1 is $1"
    if [ $1 -eq 7 ]; then
        date;
    else
        date -d "+7 hours"; 
    fi
}

wait_gpu() {
    gb=$1
    gb=$(expr $gb '*' 1024) 
    # 20480 == 20GB
    threshold=$gb
    while true; do
        mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1}' | sort -n | tail -1)
        echo "GPU= $mem MB; wait --> $gb MB|$(dateLocal $2) Thailand";
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

pilot(){
    # should be 4x1x_350 4x1x_550 10x1x_350 10x1x_550
    tr 4x1x_550 GBlr4b10_0 "-pi -se 100 -b 10 -lr -4 -s 700"
    tr 10x1x_350 GMlr4b10_0 "-pi -se 100 -b 10 -lr -4 -s 700"
    tr 4x1x_350 GClr4b10_0 "-pi -se 100 -b 10 -lr -4 -s 700"
    tr 10x1x_550 GClr4b10_0 "-pi -se 100 -b 10 -lr -4 -s 700"
}

