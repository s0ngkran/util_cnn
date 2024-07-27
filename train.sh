echo
echo "start train..."
echo
echo "python train.py"

tr(){
    config=$3
    for i in $(seq $1 $2)
        do
            python train.py $config$i --config $config -lr -4;
        done
}

tr_(){
    i=$1
    config=$2
    python train.py $config_$i --config $config -lr -4;
}

#  -> static model size| change input sigma
# sigma size          -> change size of sigma on all             | tr_720, tr_512, tr_256, tr_128, tr_64
# curriculum learning -> change on each stage of model big2small | tr_p4, tr_p6, tr_p8

tr_ 0 s64;