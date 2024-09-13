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
    i=0
    batch=10

    wait_gpu 9;
    tr ${t}y FMlr4b10_${i} "-b ${batch} -lr -4 -s 400" ;
    sleep 100;
    echo "auto run test()" >> acc;
    te ${t}y_FMlr4b10_${i} "360 -d cuda -b ${batch}";
}

# 1 4 7 10 13
t=1
i=0
batch=10
# tr ${t}y FMlr4b10_${i} "-b ${batch} -lr -4 -s 400";
te ${t}y_FMlr4b10_${i} "360 -d cuda -b ${batch}";
tr_te 4;
tr_te 7;
tr_te 10;
tr_te 13;

