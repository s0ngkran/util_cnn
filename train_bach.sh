echo
echo "start train..."
echo

source $(dirname $0)/train_util.sh;

# i=0
# tr s64 AB${i}_lr1 "-lr -1 -s 60";


# for i in {1..5}
# do
#     wait_gpu 9 7;
#     tr s256 BBlr1_${i} "-lr -1 -s 60"&
#     sleep 300;
# done


# for i in {0..4}
# do
#     wait_gpu 4 7;
#     tr s128 BBlr1_${i} "-lr -1 -s 80"&
#     sleep 300;
# done


# for i in {0..4}
# do
#     wait_gpu 2 7;
#     tr s64 BBlr1_${i} "-lr -1 -s 80"&
#     sleep 300;
# done


# for i in 5 0 1 2
# do
#      wait_gpu 6 7;
#      tr s256 BBlr1_${i} "-lr -1 -col"&
#      sleep 300;
# done


# wait_gpu 9 7;
# i=0
# tr 4x CBlr0_${i} "-nlr -4 -s 400 -b 2 -col"&
# sleep 300;

# wait_gpu 9 7;
# tr 1x CBlr4_${i} "-lr -4 -s 400 -b 2"&
# sleep 300;

# wait_gpu 9 7;
# tr 2x CBlr4_${i} "-lr -4 -s 400 -b 2"&
# sleep 300;

# wait_gpu 9 7;
# tr 3x CBlr4_${i} "-lr -4 -s 400 -b 2"&
# sleep 300;

# wait_gpu 9 7;
# i=0
# tr 6x CBlr4_${i} "-lr -4 -s 400 -b 2";
# te 6x_CBlr4_0 "720 -d cuda -b 2";

# wait_gpu 8 7;
# i=0
# tr 10x CBlr4_${i} "-lr -4 -s 350 -b 2";
# te 10x_CBlr4_0 "720 -d cuda -b 2";


# running
# i=0
# tr 8p EBlr4_${i} "-lr -4 -s 350 -b 2";
# te 8p EBlr4_${i} "360 -d cuda -b 2";
# tr 2p EBlr4_${i} "-lr -4 -s 350 -b 2";
# te 2p EBlr4_${i} "360 -d cuda -b 2";


# i=0
# tr 6p EBlr4_${i} "-lr -4 -s 350 -b 2";
# te 6p EBlr4_${i} "360 -d cuda -b 2";
# tr 4p EBlr4_${i} "-lr -4 -s 350 -b 2";
# te 4p EBlr4_${i} "360 -d cuda -b 2";


# i=0
# te 8p_EBlr4_${i} "720 -d cuda -b 2"&
# te 6p_EBlr4_${i} "128 -d cuda -b 2"&

# tr 1p EBlr4_${i} "-lr -4 -s 450 -b 2";


# 1p_EBlr4_0.best
# 2p_EBlr4_EBlr4_0.best
# 6p_EBlr4_0.best
# 8p_EBlr4_0.best

# wait_gpu 9 7;
# i=0
# tr 1p EBlr4_${i} "-lr -4 -s 450 -b 10 -col";
# te 1p_EBlr4_${i} "360 -d cuda -b 2";
# tr 2p EBlr4_EBlr4_${i} "-lr -4 -s 450 -b 10 -col";
# te 2p_EBlr4_EBlr4_${i} "360 -d cuda -b 10";
# tr 4p EBlr4_${i} "-lr -4 -s 450 -b 10 -col";
# te 4p_EBlr4_${i} "360 -d cuda -b 10";

# te 2p_EBlr4_EBlr4_0 "360 -d cpu -b 5";

# tr 4x1x_550 GBlr4b10_0 "-pi -se 100 -b 10 -lr -4 -s 700";
# te 4x1x_550_GBlr4b10_0 "360 -d cuda -b 10";


# tr 4x1x_550 HB_0 "-pi2 95 -b 10 -lr -4 -s 900 -w 4x1x_550_GBlr4b10_0.500";


    # for ep in 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300; do
    #     wait_te 4x1x_550_GBlr4b10_0 $ep ;
    # done


# for ep in {530..1000..20}
# do
#     echo "wait ep$ep...";
#     wait_te 4x1x_550_HB_0 $ep ;
# done


# tr 4x1x_550 H2B_0 "-pi2 95 -b 10 -lr -4 -s 900 -w 4x1x_550_GBlr4b10_0.500"&

# for ep in {550..1600..50}
# do
#     echo "wait ep$ep...";
#     wait_te 4x1x_550_H2B_0 $ep ;
# done

# args="-b 10 -lr -4 -s 900 -se 100"
# tr_te_360 10y FB_0 "--batch_size 10 -lr -4 -s 900 -se 100";


# args='-b 10 -lr -4 -s 900 -se 100'
# tr_te_360 4y FB_0 "$args";
# tr_te_360 7y FB_0 "$args";
# tr_te_360 13y FB_0 "$args";
# tr_te_360 10y FB_0 "$args";
# tr_te_360 1y FB_0 "$args";


# for ep in {100..1200..100}
# do
#     config=4y
#     name=FB_0
#     fname="${config}_${name}.$ep"
#     python test.py $fname 360 -d cuda -b 10 --pred_keypoints --weight save/$fname;
# done


# for ep in {100..500..100}
# do
#     config=4x1x
#     name=550_GBlr4b10_0
#     fname="${config}_${name}.$ep"
#     python test.py $fname 360 -d cuda -b 10 --pred_keypoints --weight save/$fname;
# done


# for config in 4y 10y
# do
#     name=FB_0
#     ep=best
#     fname="${config}_${name}.$ep"
#     python test.py $fname 360 -d cuda -b 5 --pred_keypoints --weight save/$fname;
# done


# for fname in s64_BBlr1_0.best s64_BBlr1_1.best s64_BBlr1_2.best s64_BBlr1_3.best s64_BBlr1_4.best
# do
#    python test.py $fname 64 -d cuda -b 10 --pred_keypoints --weight save/$fname;
# done


# for fname in s128_BBlr1_0.best s128_BBlr1_1.best s128_BBlr1_2.best s128_BBlr1_3.best s128_BBlr1_4.best
# do
#    python test.py $fname 128 -d cuda -b 10 --pred_keypoints --weight save/$fname;
# done


# for fname in s256_BBlr1_0.best s256_BBlr1_1.best s256_BBlr1_2.best s256_BBlr1_3.best s256_BBlr1_4.best s256_BBlr1_5.best
# do
#    python test.py $fname 256 -d cuda -b 10 --pred_keypoints --weight save/$fname;
# done


# fname=7y_FB_0.best
# python test.py $fname 360 -d cpu -b 10 --pred_keypoints --weight save/$fname;



args="-b 10 -lr -4 -se 100 -fs 2000 -s 2000";
# tr_te_360 10y FB_1 "$args";
# tr_te_360 7y FB_1 "$args";
# tr_te_360 4y FB_1 "$args";
# tr_te_360 10y FB_2 "$args -col";
# tr_te_360 7y FB_2 "$args";
# tr_te_360 10y FB_3 "$args";
# tr_te_360 13y FB_1 "$args";
# tr_te_360 m2 KBB_0 "$args";
# tr_te_360 1y FB_10 "$args"; # can continue
tr_te_360 1y FB_11 "$args";
tr_te_360 1y FB_12 "$args";
tr_te_360 4y FB_10 "$args";
tr_te_360 10y FB_10 "$args";
tr_te_360 13y FB_10 "$args";
tr_te_360 13y FB_11 "$args";


# for fname in m2_KBB_0.best
# do
#  python test.py $fname 360 -d cuda -b 10 --pred_keypoints --weight save/$fname -cus;
# done



