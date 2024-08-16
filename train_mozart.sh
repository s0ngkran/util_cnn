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


for i in {0..4}
do
    wait_gpu 2 
    tr s64 BMlr1_${i} "-lr -1 -s 70"
    echo sleep300
    sleep 300;
done
