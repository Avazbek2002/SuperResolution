import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import torch.distributed as dist
import os
import gc
import numpy as np
import random

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_val(rank, world_size, args, opt):
    print(f"train_val function is called with rank: {rank}")
    # logging
    
    
    torch.backends.cudnn.enabled = True
    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if rank == 0:
    	Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    	Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    	logger = logging.getLogger('base')
    	logger.info(Logger.dict2str(opt))
    	tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # dataset
    setup(rank, world_size)
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader, sampler = Data.create_dataloader(
                train_set, dataset_opt, phase, world_size, rank)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader, sampler = Data.create_dataloader(
                val_set, dataset_opt, phase, world_size, rank)
    if rank == 0:
    	logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt,rank)
    
    print(f"Finished creating the model rank {rank}")
    	
    	
    if rank == 0:
    	logger.info('Initial Model Finished')
    

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state'] and rank == 0:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
        
    if rank == 0:

    	# Initialize WandbLogger
    	import wandb
    	wandb_logger = WandbLogger(opt)
    	wandb.define_metric('validation/val_step')
    	wandb.define_metric('epoch')
    	wandb.define_metric("validation/*", step_metric="val_step")
    	print("Started logging with wandb")
    	val_step = 0
    	
    if opt['phase'] == 'train':
          # Initialize distributed training
        print("Distributed training initialized with rank:", rank)
        while current_step < n_iter:
            current_epoch += 1
            sampler.set_epoch(current_epoch)
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data, rank)
                # diffusion.optimize_parameters(current_step, accumulation_steps=opt['train']['accumulation_steps'])
                diffusion.optimize_parameters()
                if current_step % opt['train']['print_freq'] == 0 and rank == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)
                
                if current_step % opt['train']['save_checkpoint_freq'] == 0 and rank == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

                # validation
                if current_step % (opt['train']['val_freq'] * 3) == 0 and rank == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()


                        if idx < 5: # Only save/log for the first 5 samples
                            sr_img = Metrics.tensor2img(visuals['SR'], out_type=np.uint16, min_max=(-1, 1))
                            hr_img = Metrics.tensor2img(visuals['HR'], out_type=np.uint16, min_max=(-1, 1))
                            lr_img = Metrics.tensor2img(visuals['LR'], out_type=np.uint16, min_max=(-1, 1))
                            fake_img = Metrics.tensor2img(visuals['INF'], out_type=np.uint16, min_max=(-1, 1))

                            Metrics.save_img(hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

                            tb_logger.add_image(
                                'Iter_{}'.format(current_step),
                                np.transpose(np.concatenate(
                                    (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                                idx)

                    avg_psnr = avg_psnr / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    
                    if rank == 0:
                    	# log
                    	logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    	logger_val = logging.getLogger('val')  # validation logger
                    	logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    	# tensorboard logger
                    	tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger and rank == 0:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                # if current_step % opt['train']['save_checkpoint_freq'] == 0:
                #     logger.info('Saving models and training states.')
                #     diffusion.save_network(current_epoch, current_step)

                #     if wandb_logger and opt['log_wandb_ckpt']:
                #         wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger and rank == 0:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        if rank == 0:
        	logger.info('End of training.')
        cleanup()  # Clean up distributed training
    elif rank == 0:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'], out_type=np.uint16, min_max=(-1, 1))
            lr_img = Metrics.tensor2img(visuals['LR'], out_type=np.uint16, min_max=(-1, 1))
            fake_img = Metrics.tensor2img(visuals['INF'], out_type=np.uint16, min_max=(-1, 1))

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter], out_type=np.uint16, min_max=(-1, 1)), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'], out_type=np.uint16, min_max=(-1, 1))  # uint8
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][-1], out_type=np.uint16, min_max=(-1, 1)), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1], out_type=np.uint16, min_max=(-1, 1)), hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1], out_type=np.uint16, min_max=(-1, 1)), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1], out_type=np.uint16, min_max=(-1, 1)), hr_img, eval_psnr, eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('--log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    world_size = torch.cuda.device_count()

    torch.multiprocessing.spawn(
        train_val,
        args=(world_size, args, opt),
        nprocs=world_size,
        join=True
    )

