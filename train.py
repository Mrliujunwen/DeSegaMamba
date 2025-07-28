import torch
from torch.utils.data import DataLoader
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
# 步骤1: 更新模型导入
from models.DSAM import DeSegaMamba

from engine import *
import os
import sys

from utils import *
from configs.config_setting_DSAM import setting_config

import warnings

warnings.filterwarnings("ignore")


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    # 根据论文细节，图片尺寸为 512x512
    # 确保您的 config.image_size 设置正确
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config

    # 步骤2: 修改模型实例化逻辑
    # 将 'vmunet' 替换为 'desegamamba' 以匹配您的新模型
    if config.network == 'desegamamba':
        model = DeSegaMamba(
            # 从配置文件中读取参数
            num_classes=model_cfg.get('num_classes', 1),
            in_chans=model_cfg.get('input_channels', 3),
            depths=model_cfg.get('depths', [2, 2, 9, 2]),
            depths_decoder=model_cfg.get('depths_decoder', [2, 9, 2, 2]),
            drop_path_rate=model_cfg.get('drop_path_rate', 0.2),
            # 根据论文和模型定义，添加新的超参数
            dw_kernel_sizes=model_cfg.get('dw_kernel_sizes', [9, 7, 5, 3]),
            dw_kernel_sizes_decoder=model_cfg.get('dw_kernel_sizes_decoder', [3, 5, 7, 9]),
        )

        # 步骤3: 实现预训练权重加载逻辑 (迁移自旧的 load_from 方法)
        load_ckpt_path = model_cfg.get('load_ckpt_path')
        if load_ckpt_path is not None and os.path.exists(load_ckpt_path):
            print(f"==============> Loading pre-trained weight from {load_ckpt_path}...")
            # 加载预训练权重文件
            checkpoint = torch.load(load_ckpt_path, map_location='cpu')

            # 通常预训练权重保存在 'model' 或 'state_dict' 键下
            pretrained_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))

            # 获取当前模型的 state_dict
            model_dict = model.state_dict()

            # 过滤掉不匹配的键 (例如最终的分类头)
            # 只保留预训练字典中，键名存在于当前模型，且尺寸匹配的权重
            filtered_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape
            }

            # 更新当前模型的 state_dict
            model_dict.update(filtered_dict)

            # 加载更新后的 state_dict
            model.load_state_dict(model_dict, strict=False)  # 使用 strict=False 允许部分加载

            print(f"Success: {len(filtered_dict)} keys matched and loaded from pre-trained model.")
            unmatched_keys = [k for k in pretrained_dict.keys() if k not in filtered_dict.keys()]
            if unmatched_keys:
                print(
                    f"Warning: {len(unmatched_keys)} keys from checkpoint were not loaded: {unmatched_keys[:5]}...")  # 只显示前5个
        else:
            print("No pre-trained checkpoint found or path not provided. Training from scratch.")

    else:
        raise Exception(f"Network '{config.network}' is not supported! Please use 'desegamamba'.")

    model = model.cuda()

    # 根据论文，输入尺寸为 512，我们使用 config.image_size 以获得更大的灵活性
    cal_params_flops(model, config.image_size, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        loss = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config
        )

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config,
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config
    main(config)
