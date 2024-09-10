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

act(){
    i=$1
    tr s360 BMlr1_${i} "-lr -1 -s 500 -col";
    sleep 100;
    te s360_BMlr1_${i} "360 -d cuda";
}


for i in 0 2
do
    wait_gpu 6 
    act $i
    echo sleep300
    sleep 300;
done
