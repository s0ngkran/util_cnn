echo
echo "start train..."
echo

source $(dirname $0)/train_util.sh


for i in {3..5}
do
    wait_gpu 10;
    python train.py s360_AB${i}_lr0 --config s360 -lr 0 -s 30&
    sleep 300;
done


