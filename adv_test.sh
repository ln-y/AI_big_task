model_path="train_logs/resnet18_pretrain_test/version_18/checkpoints/resnet18_pretrain_test-epoch=497-val_acc=0.99.ckpt"

num=0 #0 means all
CUDA_VISIBLE_DEVICES=3
noises=("gauss" "salt")
attacks=("new_fgsm") #"bim" "pgd" "c_w" "fgsm_pgd")


export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

for eps in 0.1
do
    ##测试噪声
    # for noise in ${noises[@]}
    # do
    #     echo python ${noise}.py -model $model_path -eps $eps -num $num
    #     python ${noise}.py -model $model_path -eps $eps -num $num
    # done

    # noises_str=$(printf "%s " ${noises[@]})
    # echo python adv_test.py -model $model_path -eps $eps --attacks $noises_str
    # python adv_test.py -model $model_path -eps $eps --attacks $noises_str


    for attack in ${attacks[@]}
    do
        echo python ${attack}.py -model $model_path -eps $eps -num $num
        python ${attack}.py -model $model_path -eps $eps -num $num
    done

    attacks_str=$(printf "%s " ${attacks[@]})
    echo python adv_test.py -model $model_path -eps $eps --attacks $attacks_str 
    python adv_test.py -model $model_path -eps $eps --attacks $attacks_str 
done

