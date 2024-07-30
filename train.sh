echo
echo "start train..."
echo

source $(dirname $0)/train_util.sh


for i in {4..5}
do
    wait_gpu 10;
    tr s360 AB${i}_lr0 "-lr 0 -s 70" &
    sleep 300;
done


