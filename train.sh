echo
echo "start train..."
echo
echo "python train.py"

tr(){
    for i in $(seq $1 $2)
        do
            python train.py paf_720$i -b 8 -lr -4;
        done
}
#  -> static model size| change input sigma
# sigma size          -> change size of sigma on all             | tr_720, tr_512, tr_256, tr_128, tr_64
# curriculum learning -> change on each stage of model big2small | tr_p4, tr_p6, tr_p8