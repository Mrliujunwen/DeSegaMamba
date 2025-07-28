from torchvision import transforms
from utils import *

from datetime import datetime


class setting_config:
    """
    the config of training setting.
    """

    network = 'desegamamba'

    model_config = {
        'num_classes': 1,
        'input_channels': 3,

        'depths': [2, 2, 9, 2],
        'depths_decoder': [2, 9, 2, 2],

        'drop_path_rate': 0.2,  # 这是一个常用的正则化值，可以根据需要调整

        # 设置预训练权重路径。如果不需要预训练，请设置为 None。
        'load_ckpt_path': '',

        'dw_kernel_sizes': [9, 7, 5, 3],
        'dw_kernel_sizes_decoder': [3, 5, 7, 9],
    }

    datasets = 'isic18'
    if datasets == 'isic18':
        data_path = './data/isic2018/'
    elif datasets == 'isic17':
        data_path = './data/isic2017/'
    else:
        raise Exception('datasets in not right!')

    criterion = BceDiceLoss(wb=1, wd=1)

    pretrained_path = './pre_trained/'
    num_classes = 1

    # "Input images were resized to 512×512"
    input_size_h = 512
    input_size_w = 512

    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 8  # 建议增加 num_workers 以加速数据加载，例如 4 或 8
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'

    # "Training involved 120 epochs ... and a batch size of 8."
    batch_size = 8
    epochs = 120

    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 1  # 建议每个 epoch 都进行一次验证，以便更好地追踪模型性能
    save_interval = 100
    threshold = 0.5

    train_transformer = transforms.Compose([
        myNormalize(datasets, train=True),
        myToTensor(),
        # 根据论文 "Implementation Details" 的描述 "augmented via random rotations and flips."
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w)
    ])
    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

    # "trained on a single NVIDIA RTX 4090 GPU ... using the Adam optimizer with an initial learning rate of 1.0E-04"
    opt = 'Adam'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                   'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01
        rho = 0.9
        eps = 1e-6
        weight_decay = 0.05
    elif opt == 'Adagrad':
        lr = 0.01
        lr_decay = 0
        eps = 1e-10
        weight_decay = 0.05
    elif opt == 'Adam':
        lr = 1e-4  # <--- 已更新为论文中指定的学习率
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.0001
        amsgrad = False
    elif opt == 'AdamW':
        lr = 1e-4  # 您也可以尝试 AdamW，它通常表现更好
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-2
        amsgrad = False
    elif opt == 'Adamax':
        lr = 2e-3
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0
    elif opt == 'ASGD':
        lr = 0.01
        lambd = 1e-4
        alpha = 0.75
        t0 = 1e6
        weight_decay = 0
    elif opt == 'RMSprop':
        lr = 1e-2
        momentum = 0
        alpha = 0.99
        eps = 1e-8
        centered = False
        weight_decay = 0
    elif opt == 'Rprop':
        lr = 1e-2
        etas = (0.5, 1.2)
        step_sizes = (1e-6, 50)
    elif opt == 'SGD':
        lr = 0.01
        momentum = 0.9
        weight_decay = 0.05
        dampening = 0
        nesterov = False

        # 学习率调度器配置，您可以选择一个，CosineAnnealingLR 是一个不错的选择
    sch = 'CosineAnnealingLR'
    if sch == 'StepLR':
        step_size = epochs // 5
        gamma = 0.5
        last_epoch = -1
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150]
        gamma = 0.1
        last_epoch = -1
    elif sch == 'ExponentialLR':
        gamma = 0.99
        last_epoch = -1
    elif sch == 'CosineAnnealingLR':
        T_max = epochs  # <--- 将 T_max 设置为总 epoch 数，使学习率在一个完整的周期内下降
        eta_min = 1e-6  # <--- 使用一个较小的 eta_min
        last_epoch = -1
    elif sch == 'ReduceLROnPlateau':
        mode = 'min'
        factor = 0.1
        patience = 10
        threshold = 0.0001
        threshold_mode = 'rel'
        cooldown = 0
        min_lr = 0
        eps = 1e-08
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50
        T_mult = 2
        eta_min = 1e-6
        last_epoch = -1
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20

