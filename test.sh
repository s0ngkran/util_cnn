echo
echo "start test..."
echo

# python test.py vgg_plain_ori_aug_tr_all$i -b 30 --out11 -d cpu;

te_w(){
    name=$1
    img_size=$2
    weight=$3
    python test.py $name $img_size --weight save/$weight.best -d cpu
}
te(){
    name=$1
    img_size=$2
    echo "test $name"
    python test.py $name $img_size -d cpu|grep acc>>acc
}

# s128_AC1.best
# s128_AC1.last
# s128_AC2.best
# s128_AC2.last
# s128_AC3.best
# s128_AC3.last
# s256_11.best
# s256_11.last
# s360_0.best
# s360_0.last
# s64_AC1.best
# s64_AC1.last
# s64_AC2.best
# s64_AC2.last
# s64_AC3.best
# s64_AC3.last
# s64_AC_T1_0.best
# s64_AC_T1_0.last
# s64_AC_T1_1.best
# s64_AC_T1_1.last
# s64_AC_T1_2.best
# s64_AC_T1_2.last
# s64_AC_T1_3.best
# s64_AC_T1_3.last
# s64_AC_T1_4.best
# s64_AC_T1_4.last
# s64_c1.best
# s64_c1.last
# s720_0.best
# s720_0.last
# test.best

te s64_AC_T1_0 64;
te s64_AC_T1_1 64;
te s64_AC_T1_2 64;
te s64_AC_T1_3 64;

