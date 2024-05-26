model_path="model/resnet18_pretrain_test-epoch=283-val_acc=0.96.ckpt"
eps=0.1
num=10 #0 means all

attacks=("fgsm" "bim" "pgd" "c_w")

for attack in ${attacks[@]}
do
    echo python ${attack}.py -model $model_path -eps $eps -num $num
    python ${attack}.py -model $model_path -eps $eps -num $num
done

attacks_str=$(printf "%s " ${attacks[@]})
echo python adv_test.py -model $model_path -eps $eps --attacks $attacks_str 
python adv_test.py -model $model_path -eps $eps --attacks $attacks_str 