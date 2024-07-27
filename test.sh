echo "start test..."
echo
echo "python train.py --test"

vgg_plain(){
    for i in $(seq $1 $2)
        do
            echo $i;
            python test.py vgg_plain_$i|grep acc >> acc;
        done
}
vgg_plain_no_drop(){
    for i in $(seq $1 $2)
        do
            echo $i;
            python test.py vgg_plain_no_drop_$i --no_drop|grep acc >> acc;
        done
}
vgg_plain_ori(){
    for i in $(seq $1 $2)
        do
            echo $i;
            python test.py vgg_plain_ori$i --no_drop --out11 --no_aug|grep acc >> acc;
        done
}
vgg_plain_oriori(){
    for i in $(seq $1 $2)
        do
            echo $i;
            python test.py vgg_plain_oriori$i --no_bn_dr --out11 --no_aug|grep acc >> acc;
        done
}

# python test.py vgg_plain_0;

# vgg_plain 0 10;
# vgg_plain_no_drop 0 10;


# i=1
# python test.py vgg_plain_no_drop_$i --no_drop;
#
# i=1
# python test.py vgg_plain_ori$i --no_drop --out11 --no_aug;
# vgg_plain_ori 0 10;

# i=0
# python test.py vgg_plain_oriori$i --no_bn_dr --out11 --no_aug;


i=00
python test.py vgg_plain_ori_aug_tr_all$i -b 30 --out11 -d cpu;