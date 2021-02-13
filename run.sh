# resnet
# dvc repro train_resnet18_128 -fs
# dvc repro train_resnet34_128 -fs
# dvc repro train_resnet50_128 -fs
# dvc repro train_resnet18_256 -fs
# dvc repro train_resnet18_512 -fs

# python pipe/train.py --train_data train_128 --test_data test_128 --fold 0 --epochs 9 --arch resnet18 --sz 128 --batch_size 128 --lr 1 --sched onecycle
# python pipe/train.py --train_data train_128 --test_data test_128 --fold 0 --epochs 10 --arch resnet18 --sz 128 --batch_size 128 --lr 1 --sched onecycle
# python pipe/train.py --train_data train_128 --test_data test_128 --fold 0 --epochs 11 --arch resnet18 --sz 128 --batch_size 128 --lr 1 --sched onecycle
# python pipe/train.py --train_data train_128 --test_data test_128 --fold 0 --epochs 12 --arch resnet18 --sz 128 --batch_size 128 --lr 1 --sched onecycle
# python pipe/train.py --train_data train_128 --test_data test_128 --fold 0 --epochs 15 --arch resnet18 --sz 128 --batch_size 128 --lr 1 --sched onecycle
# python pipe/train.py --train_data train_128 --test_data test_128 --fold 0 --epochs 20 --arch resnet18 --sz 128 --batch_size 128 --lr 1 --sched onecycle
# python pipe/train.py --train_data train_128 --test_data test_128 --fold 0 --epochs 9 --arch resnet18 --sz 128 --batch_size 256 --lr 1 --sched onecycle
# python pipe/train.py --train_data train_128 --test_data test_128 --fold 0 --epochs 10 --arch resnet18 --sz 128 --batch_size 256 --lr 1 --sched onecycle
# python pipe/train.py --train_data train_128 --test_data test_128 --fold 0 --epochs 11 --arch resnet18 --sz 128 --batch_size 256 --lr 1 --sched onecycle
# python pipe/train.py --train_data train_128 --test_data test_128 --fold 0 --epochs 12 --arch resnet18 --sz 128 --batch_size 256 --lr 1 --sched onecycle
# python pipe/train.py --train_data train_128 --test_data test_128 --fold 0 --epochs 15 --arch resnet18 --sz 128 --batch_size 256 --lr 1 --sched onecycle
# python pipe/train.py --train_data train_128 --test_data test_128 --fold 0 --epochs 20 --arch resnet18 --sz 128 --batch_size 256 --lr 1 --sched onecycle
# efficientnet


# resnext

# TODO:
# add augmentation
# increase image size
# use better architectures