model_path="ed1.pth"

num=0 #0 means all
j=6
CUDA_VISIBLE_DEVICES=2
noises=("gauss" "salt")
attacks=( "bim" "pgd" "cw" "fgsm")


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
        echo python ${attack}_m.py -model $model_path -eps $eps -num $num -j $j
        python ${attack}_m.py -model $model_path -eps $eps -num $num -j $j
    done

    attacks_str=$(printf "%s " ${attacks[@]})
    echo python adv_test.py -model $model_path -eps $eps --attacks $attacks_str 
    python adv_test.py -model $model_path -eps $eps --attacks $attacks_str 
done

