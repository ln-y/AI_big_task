CUDA_VISIBLE_DEVICES=2

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

for model in {"ed2_1_acc.pth","ed1_2.pth"}
do
    echo python test6.py -model $model
    python test6.py -model $model
done