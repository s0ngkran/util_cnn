echo
echo "start train..."
echo

source $(dirname $0)/train_util.sh;

# i=0
# tr s64 AB${i}_lr1 "-lr -1 -s 60";


for i in {0..4}
do
    wait_gpu 9 7;
    tr s256 BBlr1_${i} "-lr -1 -s 30"&
    sleep 300;
done


