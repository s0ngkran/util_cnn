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

# python train.py vgg_plain_0 -b 8 -lr -4


tr 1 2&
tr 3 4&
tr 5 6&
tr 7 8&
tr 9 10&
