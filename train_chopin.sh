echo
echo "start train..."
echo

source $(dirname $0)/train_util.sh

tr_te(){
    t=$1
    i=0
    batch=10

    wait_gpu 9;
    tr ${t}y FClr4b10_${i} "-b ${batch} -lr -4 -s 400" ;
    sleep 100;
    echo "auto run test()" >> acc;
    te ${t}y_FClr4b10_${i} "360 -d cuda -b ${batch}";
}



# for i in 0 3 4
# do
#     wait_gpu 30 7;
#     tr_te $i&
#     sleep 300;
# done

# i=4
# te s720_BClr1b10_${i} "720 -d cuda -b 10";
# i=4
# te s720_BClr1b10_${i} "720 -d cuda -b 10";


# tr_te 4;
# i=4
# te s720_BClr1b10_${i} "720 -d cuda -b 10";

# tr s360 BMlr1_4 "-b 10 -nlr -4 -s 400 -col" ;
# echo "auto test" >>acc;
# te s360_BMlr1_4 "360 -d cuda -b 10";

# 1 4 7 10 13
# tr_te 13;
# tr_te 10;
# tr_te 7;
# tr_te 4;
# tr_te 1;


# tr 4x1x_350 GClr4b10_0 "-pi -se 100 -b 10 -lr -4 -s 700"&
# tr 10x1x_550 GClr4b10_0 "-pi -se 100 -b 10 -lr -4 -s 700"&

# for ep in 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300; do
#     wait_te 4x1x_350_GClr4b10_0 $ep ;
#     wait_te 10x1x_550_GClr4b10_0 $ep ;
# done


# tr try try "-pi -se 100 -b 10 -lr -4 -s 700";
# tr 10x1x_550 HC_0 "-pi2 95 -b 10 -lr -4 -s 900 -w 10x1x_550_GClr4b10_0.500";


# for ep in {530..1000..20}
# do
#     echo "wait ep$ep...";
#     wait_te 10x1x_550_HC_0 $ep;
# done


# tr 10x1x_550 H2C_0 "-pi2 95 -b 10 -lr -4 -s 900 -w 10x1x_550_GClr4b10_0.500"&

# for ep in {550..1600..50}
# do
#     echo "wait ep$ep...";
#     wait_te 10x1x_550_H2C_0 $ep ;
# done

tr_te_bi(){
    config=$1
    name=$2

    BATCH=10
    wait_gpu 9;
    tr ${config} ${name} "-b ${BATCH} -lr -4 -s 700 -se 100";
    echo "done $config ${name}";
    # sleep 100;
    # te ${config}_${name} "360 -d cuda -b ${BATCH}";
    # echo "^auto run test()" >> acc;
}

# tr_te_bi bi10 IC_0&
# tr_te_bi bi90 IC_0&
# tr_te_bi bi50 IC_0&
# tr_te_bi bi10 IC_1&
# tr_te_bi bi10 IC_2&
# wait
# echo "done all 10 90 50";

# for ep in {300..1000..100}
# do
#     echo "wait ep$ep...";
#     wait_te bi10_IC_0 $ep;
# done


# config=4y
# name=FC_0
# BATCH=10
# tr ${config} ${name} "-b ${BATCH} -lr -4 -s 700 -se 100";



# for ep in {900..1300..100}
# do
#     config=bi50
#     name=IC_0
#     fname="${config}_${name}.$ep"
#     python test.py $fname 360 -d cuda -b 10 --pred_keypoints --weight save/$fname;
# done

# for ep in {900..1300..100}
#     python test.py $fname 360 -d cuda -b 10 --pred_keypoints --weight save/$fname;
# done
# for ep in {100..700..100}
# do
#     config=bi10
#     name=IC_1
#     fname="${config}_${name}.$ep"
#     python test.py $fname 360 -d cuda -b 10 --pred_keypoints --weight save/$fname;
# done
# for ep in {100..400..100}
# do
#     config=bi10
#     name=IC_2
#     fname="${config}_${name}.$ep"
#     python test.py $fname 360 -d cuda -b 10 --pred_keypoints --weight save/$fname;
# done


# for ep in {100..500..100}
# do
#     config=10x1x_550
#     name=GClr4b10_0
#     fname="${config}_${name}.$ep"
#     python test.py $fname 360 -d cuda -b 5 --pred_keypoints --weight save/$fname;
# done


# s360_BMlr1_4.best
# for fname in s720_BClr1b10_0.best s720_BClr1b10_1.best s720_BClr1b10_2.best s720_BClr1b10_3.best s720_BClr1b10_4.best
# do
#     python test.py $fname 720 -d cuda -b 2 --pred_keypoints --weight save/$fname;
# done


# for fname in 4y_FC_0.100 4y_FC_0.1000 4y_FC_0.1100 4y_FC_0.200 4y_FC_0.300 4y_FC_0.400 4y_FC_0.500 4y_FC_0.600 4y_FC_0.700 4y_FC_0.800 4y_FC_0.900 4y_FC_0.best bi10_IC_1.100 bi10_IC_1.200 bi10_IC_1.300 bi10_IC_1.400 bi10_IC_1.500 bi10_IC_1.600 bi10_IC_1.700 bi10_IC_1.800 bi10_IC_1.900 bi10_IC_1.best bi10_IC_2.100 bi10_IC_2.1000 bi10_IC_2.1100 bi10_IC_2.1200 bi10_IC_2.1300 bi10_IC_2.1400 bi10_IC_2.1500 bi10_IC_2.200 bi10_IC_2.300 bi10_IC_2.400 bi10_IC_2.500 bi10_IC_2.600 bi10_IC_2.700 bi10_IC_2.800 bi10_IC_2.900 bi10_IC_2.best
# do
#     echo $fname;
#     python test.py $fname 360 -d cuda -b 20 --pred_keypoints --weight save/$fname;
# done


config="m1"
name="JC_0"
# wait_gpu 25;
# tr ${config} ${name} "-cus -fs 1500"&


# tr m1 JC_0 "-col -cus -b 10 -lr -4 -fs 1500 -se 100"&
# tr m2 JC_0 "-col -cus -b 10 -lr -4 -fs 1500 -se 100"&
# tr m3 JC_0 "-col -cus -b 10 -lr -4 -fs 1500 -se 100"&

test_m(){
    config=$1
    name=$2
    ep=$3
    fname="${config}_$name.$ep"
    python test.py $fname 360 -d cuda -b 20 --pred_keypoints -cus --weight save/$fname;

    # how to use
    # test_m m1 JC_0 best
}

test_m_fname(){
    fname=$1
    echo "testing... $fname"
    python test.py $fname 360 -d cuda -b 30 --pred_keypoints -cus --weight save/$fname;
}
test_no_cus(){
    fname=$1
    echo "testing... $fname"
    python test.py $fname 360 -d cuda -b 30 --pred_keypoints --weight save/$fname;
}

# test_m m1 JC_0 best;
# test_m m2 JC_0 best;
# test_m m3 JC_0 best;
a(){
    for fname in m1_JC_0.1000 m1_JC_0.1100 m1_JC_0.1200 m1_JC_0.1300 m1_JC_0.1400 m1_JC_0.1500 m1_JC_0.1600 m1_JC_0.1700 m1_JC_0.1800 m1_JC_0.200 m1_JC_0.300 m1_JC_0.400 m1_JC_0.500 m1_JC_0.600 m1_JC_0.700 m1_JC_0.800 m1_JC_0.900 m1_JC_0.best
    do
        test_m_fname "$fname"
    done
}


# tr m2 KC_0 "-cus -b 20 -lr -4 -fs 1500 -se 100"&
# tr n1 LC_0 "-b 20 -lr -4 -fs 1500 -se 100"&
# tr n1 LC_1 "-b 20 -lr -4 -fs 1500 -se 100";
# tr n1 LC_2 "-b 20 -lr -4 -fs 1500 -se 100";
# tr n1 LC_3 "-b 20 -lr -4 -fs 1500 -se 100";
tr m2 KCC_0 "-cus -b 20 -lr -4 -fs 1500 -se 100";

# test_m_fname m2_KC_0.best
# test_no_cus n1_LC_0.best
 
# for fname in n1_LC_0.100 n1_LC_0.1000 n1_LC_0.1100 n1_LC_0.1200 n1_LC_0.1300 n1_LC_0.1400 n1_LC_0.1500 n1_LC_0.1600 n1_LC_0.200 n1_LC_0.300 n1_LC_0.400 n1_LC_0.500 n1_LC_0.600 n1_LC_0.700 n1_LC_0.800 n1_LC_0.900
# do
#     echo $fname
#     test_no_cus "$fname"
# done

