echo
echo "start train..."
echo

source $(dirname $0)/train_util.sh


for i in {0..4}
do
    wait_gpu 10;
    tr s720 AB${i}_lr0 "-lr -0 -s 60" &
    sleep 300;
done


