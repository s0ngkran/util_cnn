echo
echo "start train..."
echo
echo "python train.py"

tr(){
    config=$1
    for i in $(seq $2 $3)
        do
            python train.py $config_$i --config $config;
        done
}

tr_(){
    config=$1
    i=$2
    python train.py $config_$i --config $config;
}

#  -> static model size| change input sigma
# sigma size          -> change size of sigma on all             | tr_720, tr_512, tr_256, tr_128, tr_64
# curriculum learning -> change on each stage of model big2small | tr_p4, tr_p6, tr_p8

# running
# tr_ s64 0&
# tr_ s64 2&

tr_ s128 0&