model_path="model/resnet18_pretrain_test-epoch=283-val_acc=0.96.ckpt"
eps=0.1
num=0 #0 means all
CUDA_VISIBLE_DEVICES=1
noises=("gauss" "salt")
attacks=("fgsm" "bim" "pgd" "c_w" "fgsm_pgd")

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
##测试噪声
for noise in ${noises[@]}
do
    echo python ${noise}.py -model $model_path -eps $eps -num $num
    python ${noise}.py -model $model_path -eps $eps -num $num
done

noises_str=$(printf "%s " ${noises[@]})
echo python adv_test.py -model $model_path -eps $eps --attacks $noises_str
python adv_test.py -model $model_path -eps $eps --attacks $noises_str

for attack in ${attacks[@]}
do
    echo python ${attack}.py -model $model_path -eps $eps -num $num
    python ${attack}.py -model $model_path -eps $eps -num $num
done

attacks_str=$(printf "%s " ${attacks[@]})
echo python adv_test.py -model $model_path -eps $eps --attacks $attacks_str 
python adv_test.py -model $model_path -eps $eps --attacks $attacks_str 