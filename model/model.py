import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
import torch.nn.utils as nn_utils
import random
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt, rank):
        super(DDPM, self).__init__(opt)
        print(f"--> Starting Rank {rank}. Distributed flag is: {opt.get('distributed')}")
        
        # define network and load pretrained models
        netG = networks.define_G(opt)
        self.rank = rank
        self.netG = self.set_device(netG, self.rank)

        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if opt.get('distributed'):
        	"assert torch.cuda.is_available()"
        	print(f"Just before DDP in rank {rank}")
        	self.netG = DDP(self.netG, device_ids=[self.rank])
        	print(f"Wrapped around DDP {rank}")
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.module.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.module.parameters())
	
            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.warmup_updates = opt['train']["optimizer"]["warmup_updates"]

            def lr_lambda(update):
                """Linear warm-up from 0 → target_lr in <warmup_updates> optimiser *updates*.
                After warm-up, keep LR constant (or plug in your own decay here)."""
                if update < self.warmup_updates:
                    return float(update) / float(max(1, self.warmup_updates))
                return 1.0   

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optG, lr_lambda)
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data, rank):
        self.data = self.set_device(data, rank)

    # def optimize_parameters(self, iter, accumulation_steps=32, max_norm=1.0, norm_type=2):
    #     l_pix = self.netG(self.data)
    #     # need to average in multi-gpu
    #     b, c, h, w = self.data['HR'].shape
    #     l_pix = l_pix.sum()/int(b*c*h*w)
        
    #     if (iter - 1) % accumulation_steps == 0:
    #         self.log_dict['l_pix'] = l_pix.item()

    #     # l_pix = l_pix / accumulation_steps
    #     l_pix.backward()
        
    #     # if iter % accumulation_steps == 0:
             
    #     #      nn_utils.clip_grad_norm_(self.netG.parameters(),
    #     #                          max_norm=max_norm,
    #     #                          norm_type=norm_type)

    #     #      self.optG.step()
    #     #      self.lr_scheduler.step()
    #     #      self.optG.zero_grad()

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], continous)
            else:
                self.SR = self.netG.super_resolution(
                    self.data['SR'], continous)
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, DDP):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, DDP):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, DDP):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()

        # Save RNG states
        opt_state['rng_state_py'] = random.getstate()
        opt_state['rng_state_np'] = np.random.get_state()
        opt_state['rng_state_torch'] = torch.get_rng_state()
        if torch.cuda.is_available():  # Only save CUDA state if CUDA is being used
            opt_state['rng_state_cuda'] = torch.cuda.get_rng_state()
            opt_state['rng_state_cuda_all'] = torch.cuda.get_rng_state_all()

        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
