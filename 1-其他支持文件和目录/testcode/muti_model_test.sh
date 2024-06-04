
versions=(20)
de_noise=0

for version in ${versions[@]}
do
    for file in train_logs/resnet18_pretrain_test/version_${version}/checkpoints/*
    do
        echo python test.py -model $file --denoise $de_noise
        python test.py -model $file --denoise $de_noise
    done
done