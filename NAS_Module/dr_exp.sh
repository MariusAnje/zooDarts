echo ===============    start trying ===========
for i in `seq 1 30`; do
    echo ==============$i ===============
    python -u dr_controller.py --model resnet18  -j 24 --data-path "/dataset/ImageNet/" --lr 0.0001 --rl --rl_optimizer rsm --hwt --alpha 0.7 --target_lat "4 5" --target_acc "70 90" --dconv "832, 1, 32, 32, 5, 6, 10, 14" --cconv "70, 36, 64, 64, 7, 18, 6, 6" -b 48 --device cuda:1
done
