echo "start test..."
echo
echo "python train.py --test"


# python test.py vgg_plain_ori_aug_tr_all$i -b 30 --out11 -d cpu;

python test.py test_save 720 --weight save/test.best -d cpu


