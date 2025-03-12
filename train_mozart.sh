echo "train..."

source $(dirname $0)/train_util.sh
echo 'after source';


# i=0
# tr s360 BMlr1_${i} "-col -lr -1 -s 70";
# echo sleep300
# sleep 300;
# wait_gpu 10;
# i=1
# tr s360 BMlr1_${i} "-col -lr -1 -s 70";
# echo sleep300
# sleep 300;

# for i in {2..6}
# do
#     wait_gpu 10 
#     tr s360 BMlr1_${i} "-lr -1 -s 70"
#     echo sleep300
#     sleep 300;
# done

# act(){
#     i=$1
#     tr s360 BMlr1_${i} "-lr -1 -s 500 -col";
#     sleep 100;
#     te s360_BMlr1_${i} "360 -d cuda";
# }


# for i in 0 2
# do
#     wait_gpu 6 
#     act $i
#     echo sleep300
#     sleep 300;
# done


# wait_gpu 9
# i=0
# tr 3x CMlr0_${i} "-col -nlr -4 -s 500 -b 2";
# echo sleep300;
# sleep 300;


# wait_gpu 9
# i=0
# tr 2x CMlr4_${i} "-lr -4 -s 500 -b 2";
# echo sleep300;
# sleep 300;

# wait_gpu 9
# i=0
# tr 4x CMlr4_${i} "-lr -4 -s 500 -b 2";
# echo sleep300;
# sleep 300;

# wait_gpu 9
# i=0
# tr 1x CMlr4_${i} "-lr -4 -s 500 -b 2";
# echo sleep300;
# sleep 300;

# wait_gpu 9
# i=0
# tr 8x CMlr4_${i} "-lr -4 -s 350 -b 2";
# echo "auto test 8x" >> acc;
# te 8x_CMlr4_${i} "720 -b 2 -d cuda";

# tr_te(){
#     i=$1
#     tr s720 BClr1b10_${i} "-b 5 -nlr -4 -s 200 -col" ;
#     sleep 100;
#     echo "auto run test()" >> acc;
#     te s720_BClr1b10_${i} "720 -d cuda -b 5";
# }
tr_te(){
    t=$1
    i=$2
    batch=10

    wait_gpu 9;
    tr ${t}y FMlr4b10_${i} "-b ${batch} -lr -4 -s 900 -se 100" ;
    # tr 4y FMlr4b10_0 "-b 10 -lr -4 -s 400" ;
    sleep 100;
    # echo "auto run test()" >> acc;
    # te ${t}y_FMlr4b10_${i} "360 -d cuda -b ${batch}";
}

# 1 4 7 10 13
# t=1
# i=0
# batch=10
# tr ${t}y FMlr4b10_${i} "-b ${batch} -lr -4 -s 400 -col";
# te ${t}y_FMlr4b10_${i} "360 -d cuda -b ${batch}";
# tr_te 1;
# tr_te 7;
# tr_te 10;
# tr_te 13;

# tr_te 4;
# tr_te 13;
# tr_te 10;

# for ep in {100..1200..100}
# do
#     config=10y
#     name=FMlr4b10_0
#     fname="${config}_${name}.$ep"
#     python test.py $fname 360 -d cpu -b 10 --pred_keypoints --weight save/$fname;
# done




# for fname in s64_BMlr1_0.best s64_BMlr1_1.best s64_BMlr1_2.best s64_BMlr1_3.best s64_BMlr1_4.best
# do
#     python test.py $fname 64 -d cuda -b 10 --pred_keypoints --weight save/$fname;
# done


# for fname in s360_BMlr1_0.best s360_BMlr1_1.best s360_BMlr1_2.best s360_BMlr1_3.best s360_BMlr1_4.best
# do
#     python test.py $fname 360 -d cuda -b 10 --pred_keypoints --weight save/$fname;
# done



# for i in 7 1 10 13 4

# tr_te 7;
# tr_te 1;
# tr_te 10;
# tr_te 13;
# tr_te 4;

# tr_te 2 0;
# tr_te 2 1;
# tr_te 2 2;
# tr_te 2 3;
# tr_te 2 4;
# tr_te 2 5;


# for fname in 1y_FMlr4b10_1.best 
# do
#     echo $fname
#     python test.py $fname 360 -d cuda -b 10 --pred_keypoints --weight save/$fname;
# done
# for fname in 2y_FMlr4b10_1.100 2y_FMlr4b10_1.1000 2y_FMlr4b10_1.1100 2y_FMlr4b10_1.1200 2y_FMlr4b10_1.1300 2y_FMlr4b10_1.1400 2y_FMlr4b10_1.1500 2y_FMlr4b10_1.1600 2y_FMlr4b10_1.200 2y_FMlr4b10_1.300 2y_FMlr4b10_1.400 2y_FMlr4b10_1.500 2y_FMlr4b10_1.600 2y_FMlr4b10_1.700 2y_FMlr4b10_1.800 2y_FMlr4b10_1.900 2y_FMlr4b10_1.best 2y_FMlr4b10_2.100 2y_FMlr4b10_2.1000 2y_FMlr4b10_2.1100 2y_FMlr4b10_2.1200 2y_FMlr4b10_2.1300 2y_FMlr4b10_2.1400 2y_FMlr4b10_2.200 2y_FMlr4b10_2.300 2y_FMlr4b10_2.400 2y_FMlr4b10_2.500 2y_FMlr4b10_2.600 2y_FMlr4b10_2.700 2y_FMlr4b10_2.800 2y_FMlr4b10_2.900 2y_FMlr4b10_2.best
# do
#     echo $fname
#     python test.py $fname 360 -d cuda -b 10 --pred_keypoints --weight save/$fname;
# done


# for fname in 2y_FMlr4b10_3.100 2y_FMlr4b10_3.1000 2y_FMlr4b10_3.1100 2y_FMlr4b10_3.1200 2y_FMlr4b10_3.1300 2y_FMlr4b10_3.1400 2y_FMlr4b10_3.1500 2y_FMlr4b10_3.1600 2y_FMlr4b10_3.1700 2y_FMlr4b10_3.1800 2y_FMlr4b10_3.1900 2y_FMlr4b10_3.200 2y_FMlr4b10_3.2000 2y_FMlr4b10_3.2100 2y_FMlr4b10_3.300 2y_FMlr4b10_3.400 2y_FMlr4b10_3.500 2y_FMlr4b10_3.600 2y_FMlr4b10_3.700 2y_FMlr4b10_3.800 2y_FMlr4b10_3.900 2y_FMlr4b10_3.best 2y_FMlr4b10_4.100 2y_FMlr4b10_4.1000 2y_FMlr4b10_4.1100 2y_FMlr4b10_4.1200 2y_FMlr4b10_4.1300 2y_FMlr4b10_4.1400 2y_FMlr4b10_4.1500 2y_FMlr4b10_4.1600 2y_FMlr4b10_4.1700 2y_FMlr4b10_4.1800 2y_FMlr4b10_4.1900 2y_FMlr4b10_4.200 2y_FMlr4b10_4.2000 2y_FMlr4b10_4.2100 2y_FMlr4b10_4.2200 2y_FMlr4b10_4.2300 2y_FMlr4b10_4.2400 2y_FMlr4b10_4.2500 2y_FMlr4b10_4.2600 2y_FMlr4b10_4.2700 2y_FMlr4b10_4.2800 2y_FMlr4b10_4.300 2y_FMlr4b10_4.400 2y_FMlr4b10_4.500 2y_FMlr4b10_4.600 2y_FMlr4b10_4.700 2y_FMlr4b10_4.800 2y_FMlr4b10_4.900 2y_FMlr4b10_4.best 2y_FMlr4b10_5.100 2y_FMlr4b10_5.1000 2y_FMlr4b10_5.1100 2y_FMlr4b10_5.1200 2y_FMlr4b10_5.1300 2y_FMlr4b10_5.1400 2y_FMlr4b10_5.1500 2y_FMlr4b10_5.1600 2y_FMlr4b10_5.1700 2y_FMlr4b10_5.1800 2y_FMlr4b10_5.1900 2y_FMlr4b10_5.200 2y_FMlr4b10_5.2000 2y_FMlr4b10_5.2100 2y_FMlr4b10_5.2200 2y_FMlr4b10_5.300 2y_FMlr4b10_5.400 2y_FMlr4b10_5.500 2y_FMlr4b10_5.600 2y_FMlr4b10_5.700 2y_FMlr4b10_5.800 2y_FMlr4b10_5.900 2y_FMlr4b10_5.best
# do
#     echo $fname
#     python test.py $fname 360 -d cuda -b 50 --pred_keypoints --weight save/$fname;
# done


args="-b 30 -lr -4 -se 100 -fs 1000 -s 999";
# tr s-heat SS_1 "$args"
# tr s-heat-256 SSB_1 "$args";
tr s-heat-scaled-256 SSC_M01 "$args";

# fname="s-heat_SS_1.best"
# fname="s-heat_SS_2.best"
# python test.py $fname 128 -d cuda -b 1 --config s-heat --pred_keypoints --weight save/$fname;
