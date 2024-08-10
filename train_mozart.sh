echo "train..."

source $(dirname $0)/train_util.sh
echo 'after source';

for i in {0..4}
do
    wait_gpu 10 true
    tr 4x _Mlr1_${i} "-lr -1 -s 70"
done