echo
echo "start train..."
echo
echo "python train.py"

tr(){
    for i in $(seq $1 $2)
        do
            python train.py vgg_plain_$i -b 8 -lr -4;
        done
}
trnd(){
    for i in $(seq $1 $2)
        do
            python train.py vgg_plain_no_drop_$i -b 8 -lr -4 --no_drop;
        done
}
trori(){
    for i in $(seq $1 $2)
        do
            python train.py vgg_plain_ori$i -b 8 -lr -4 --no_drop --out11 --no_aug -s 50;
        done
}
troriori(){
    for i in $(seq $1 $2)
        do
            python train.py vgg_plain_oriori$i -b 8 -lr -4 --no_bn_dr --out11 --no_aug -s 50;
        done
}
trall(){
    for i in $(seq $1 $2)
        do
            python train.py vgg_plain_ori_aug_tr_all$i --train_all -b 8 -lr -4 --out11 -s 50;
        done
}

# python train.py vgg_plain_0 -b 8 -lr -4
# 

# tr => out30, with dropout, b8, lr4,
# trnd => out30, wo/ dropout, b8, lr4
# trori => out11, no_aug, wo/ dropout, b8, lr4


# tr 1 2&
# tr 3 4&
# tr 5 6&
# tr 7 8&
# tr 9 10&

# echo 'trnd'
# trnd 1 2&
# trnd 3 4&
# trnd 5 6&
# trnd 7 8&
# trnd 9 10&

# i=0
# python train.py vgg_plain_ori$i -b 8 -lr -4 --no_drop --out11 --no_aug;
#
# echo 'tr ori'
# trori 1 2&
# trori 3 4&
# trori 5 6&
# trori 7 8&
# trori 9 10&

# echo 'tr oriori'
# troriori 1 2&
# troriori 3 4&
# troriori 5 6&
# troriori 7 8&
# troriori 9 10&
# i=0
# python train.py vgg_plain_oriori$i -b 8 -lr -4 --no_bn_dr --out11 --no_aug -s 50;

# i=0
# python train.py vgg_plain_oriori_aug$i -b 8 -lr -4 --no_bn_dr --out11 -s 50;

echo trall
trall 1 2& # running
trall 3 4& # running
trall 5 6&
trall 7 8&
trall 9 10&