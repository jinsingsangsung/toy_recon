import imageio
# from pygifsicle import optimize
import argparse
import os
import random
import shutil
from datetime import datetime
import numpy as np
import csv
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from model_all import Cifar, Cifar1D, SimpleConv, HNeRVDecoder, TransformInput #S4NDConv, S4NDPure, PureTransformer, HiPPOConvPure, S4Pure, S5Pure, S4DPure, SSMConv, MySSMConv
# from model_variants import VariantsA, VariantsB, VariantsC, Vanilla
from model_variants_1d import VariantsA, VariantsB, VariantsC, Vanilla
# from cifar import Cifar
from hnerv_utils import *
from torch.utils.data import Subset
from copy import deepcopy
from dahuffman import HuffmanCodec
from torchvision.utils import save_image
import pandas as pd
from lion_pytorch import Lion
from einops import rearrange, repeat
import torchvision
from dct import SimpleNetEncoder, SimpleNetDecoder

def main():
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='data/cifar-100-python/', help='data path for vid')
    parser.add_argument('--vid', type=str, default='bunny', help='video id',)
    parser.add_argument('--shuffle_data', action='store_true', help='randomly shuffle the frame idx')
    parser.add_argument('--data_split', type=str, default='1_1_1', 
        help='Valid_train/total_train/all data split, e.g., 18_19_20 means for every 20 samples, the first 19 samples is full train set, and the first 18 samples is chose currently')
    parser.add_argument('--crop_list', type=str, default='640_1280', help='video crop size',)
    parser.add_argument('--resize_list', type=str, default='-1', help='video resize size',)
    parser.add_argument('--clip_len', type=int, default=1, help='video resize size',)

    # NERV architecture parameters
    # Embedding and encoding parameters
    parser.add_argument('--embed', type=str, default='', help='empty string for HNeRV, and base value/embed_length for NeRV position encoding')
    parser.add_argument('--ks', type=str, default='0_1_5', help='kernel size for encoder and decoder')
    parser.add_argument('--enc_strds', type=int, nargs='+', default=[5,4,4,2,2], help='stride list for encoder')
    parser.add_argument('--enc_dim', type=str, default='64_16', help='enc latent dim and embedding ratio')
    parser.add_argument('--modelsize', type=float,  default=1.5, help='model parameters size: model size + embedding parameters')
    parser.add_argument('--saturate_stages', type=int, default=-1, help='saturate stages for model size computation')

    # Decoding parameters: FC + Conv
    parser.add_argument('--fc_hw', type=str, default='9_16', help='out size (h,w) for mlp')
    parser.add_argument('--reduce', type=float, default=1.2, help='chanel reduction for next stage')
    parser.add_argument('--lower_width', type=int, default=12, help='lowest channel width for output feature maps')
    parser.add_argument('--dec_strds', type=int, nargs='+', default=[5, 4, 4, 2, 2], help='strides list for decoder')
    parser.add_argument('--num_blks', type=str, default='1_1', help='block number for encoder and decoder')
    parser.add_argument("--conv_type", default=['convnext', 'pshuffel'], type=str, nargs="+",
        help='conv type for encoder/decoder', choices=['pshuffel', 'conv', 'convnext', 'interpolate'])
    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='swish', help='activation to use', 
        choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])

    # General training setups
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--start_epoch', type=int, default=-1, help='starting epoch')
    parser.add_argument('--not_resume', action='store_true', help='not resume from latest checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=300, help='Epoch number')
    parser.add_argument('--block_params', type=str, default='1_1', help='residual blocks and percentile to save')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.0002')
    parser.add_argument('--lr_type', type=str, default='cosine_0.1_1_0.1', help='learning rate type, default=cosine')
    parser.add_argument('--loss', type=str, default='L2', help='loss type, default=L2')
    parser.add_argument('--out_bias', default='tanh', type=str, help='using sigmoid/tanh/0.5 for output prediction')

    # evaluation parameters
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--eval_freq', type=int, default=30, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--quant_model_bit', type=int, default=8, help='bit length for model quantization')
    parser.add_argument('--quant_embed_bit', type=int, default=6, help='bit length for embedding quantization')
    parser.add_argument('--quant_axis', type=int, default=0, help='quantization axis (-1 means per tensor)')
    parser.add_argument('--dump_images', action='store_true', default=False, help='dump the prediction images')
    parser.add_argument('--dump_videos', action='store_true', default=False, help='concat the prediction images into video')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')
    parser.add_argument('--encoder_file',  default='', type=str, help='specify the embedding file')

    # distribute learning parameters
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training,  added to suffix!!!!')

    # logging, output directory, 
    parser.add_argument('--debug', action='store_true', help='defbug status, earlier for train/eval')  
    parser.add_argument('-p', '--print-freq', default=100, type=int,)
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='unify', help='folder to output images and model checkpoints')
    parser.add_argument('--suffix', default='', help="suffix str for outf")
    
    parser.add_argument('--ms_mamba', action='store_true', help='apply multiscale mamba')
    parser.add_argument('--dataset_length', type=int, default=1000)
    parser.add_argument('--model', default='base', help="model type", choices=['base', 's4', 's4d', 'mamba', 's4nd', 'hippo', 'transformer', 's3k', 's5'])
    parser.add_argument('--dct', action='store_true', help='output dct to see if model can embed signal info')
    parser.add_argument('--variant', type=str, default='a', help='variant of the model', choices=['a', 'b', 'c', 'd', 'v'])

    args = parser.parse_args()
    torch.set_printoptions(precision=4) 
    if args.debug:
        args.eval_freq = 1
        args.outf = 'output/debug'
    else:
        args.outf = os.path.join('output', args.outf)

    args.enc_strd_str, args.dec_strd_str = ','.join([str(x) for x in args.enc_strds]), ','.join([str(x) for x in args.dec_strds])
    extra_str = 'Size{}_ENC_{}_{}_DEC_{}_{}_{}{}{}'.format(args.modelsize, args.conv_type[0], args.enc_strd_str, 
        args.conv_type[1], args.dec_strd_str, '' if args.norm == 'none' else f'_{args.norm}', 
        '_dist' if args.distributed else '', '_shuffle_data' if args.shuffle_data else '',)
    args.quant_str = f'quant_M{args.quant_model_bit}_E{args.quant_embed_bit}'
    embed_str = f'{args.embed}_Dim{args.enc_dim}'
    exp_id = f'{args.vid}/{args.data_split}_{embed_str}_FC{args.fc_hw}_KS{args.ks}_RED{args.reduce}_low{args.lower_width}_blk{args.num_blks}' + \
            f'_e{args.epochs}_b{args.batchSize}_{args.quant_str}_lr{args.lr}_{args.lr_type}_{args.loss}_{extra_str}{args.act}{args.block_params}{args.suffix}'
    args.exp_id = exp_id

    args.outf = os.path.join(args.outf, exp_id)
    if args.overwrite and os.path.isdir(args.outf):
        print('Will overwrite the existing output dir!')
        shutil.rmtree(args.outf)

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    port = hash(args.exp_id) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    torch.set_printoptions(precision=2) 
    args.ngpus_per_node = torch.cuda.device_count()
    if args.distributed and args.ngpus_per_node > 1:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        train(None, args)

def data_to_gpu(x, device):
    return x.to(device)

def train(local_rank, args):
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    if args.distributed and args.ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.ngpus_per_node,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()        
        args.batchSize = int(args.batchSize / args.ngpus_per_node)

    args.metric_names = ['pred_seen_psnr', 'pred_seen_ssim', 'pred_unseen_psnr', 'pred_unseen_ssim',
        'quant_seen_psnr', 'quant_seen_ssim', 'quant_unseen_psnr', 'quant_unseen_ssim']
    best_metric_list = [torch.tensor(0) for _ in range(len(args.metric_names))]

    # setup dataloader    
    # dataset = Cifar(args)
    dataset = Cifar1D(args)
    torch.manual_seed(42)
    # num_tensors = 100
    # tensor_shape = (3, 32, 32)
    # dataset = [torch.rand(tensor_shape) for _ in range(num_tensors)]
    # save_image(dataset[0], 'test.png')
    args.dump_vis = (args.dump_images or args.dump_videos)
    args.final_size=0
    # Compute the parameter number
    if 'pe' in args.embed or 'le' in args.embed:
        embed_param = 0
        embed_dim = int(args.embed.split('_')[-1]) * 2
        fc_param = np.prod([int(x) for x in args.fc_hw.split('_')])
    else:
        total_enc_strds = np.prod(args.enc_strds)
        embed_hw = args.final_size / total_enc_strds**2
        enc_dim1, embed_ratio = [float(x) for x in args.enc_dim.split('_')]
        embed_dim = int(embed_ratio * args.modelsize * 1e6 / 1 / embed_hw) if embed_ratio < 1 else int(embed_ratio) 
        embed_param = float(embed_dim) / total_enc_strds**2
        args.enc_dim = f'{int(enc_dim1)}_{embed_dim}' 
        fc_param = (np.prod(args.enc_strds) // np.prod(args.dec_strds))**2 * 9

    decoder_size = args.modelsize * 1e6 - embed_param
    ch_reduce = 1. / args.reduce
    dec_ks1, dec_ks2 = [int(x) for x in args.ks.split('_')[1:]]
    fix_ch_stages = len(args.dec_strds) if args.saturate_stages == -1 else args.saturate_stages
    a =  ch_reduce * sum([ch_reduce**(2*i) * s**2 * min((2*i + dec_ks1), dec_ks2)**2 for i,s in enumerate(args.dec_strds[:fix_ch_stages])])
    b =  embed_dim * fc_param 
    c =  args.lower_width **2 * sum([s**2 * min(2*(fix_ch_stages + i) + dec_ks1, dec_ks2)  **2 for i, s in enumerate(args.dec_strds[fix_ch_stages:])])
    args.fc_dim = int(np.roots([a,b,c - decoder_size]).max())


    ##### get model params and flops #####
    if args.variant == "a":
        model = VariantsA(model=args.model, in_channels=3, out_channels=8, kernel_size=8)
    elif args.variant == "b":
        model = VariantsB(model=args.model, in_channels=3, out_channels=8, kernel_size=8)
    elif args.variant == "c": # or args.variant == "d":
        model = VariantsB(model=args.model, in_channels=3, out_channels=8, kernel_size=8)
    else:
        print("working on Vanilla model")
        model = Vanilla(model=args.model, in_channels=3, out_channels=8, kernel_size=8)

    if local_rank in [0, None]:
        encoder_param = (sum([p.data.nelement() for p in model.encoder.parameters()]) / 1e6) 
        decoder_param = (sum([p.data.nelement() for p in model.decoder.parameters()]) / 1e6) 
        total_param = decoder_param + embed_param / 1e6
        args.encoder_param, args.decoder_param, args.total_param = encoder_param, decoder_param, total_param
        param_str = f'Encoder_{round(encoder_param, 2)}M_Decoder_{round(decoder_param, 2)}M_Total_{round(total_param, 2)}M'
        # print(f'{args}\n {model}\n {param_str}', flush=True)
        print(f'{param_str}', flush=True)
        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
            f.write(str(model) + '\n' + f'{param_str}\n')
        writer = SummaryWriter(os.path.join(args.outf, param_str, 'tensorboard'))
    else:
        writer = None

    # distrite model to gpu or parallel
    print("Use GPU: {} for training".format(local_rank))
    # if args.distributed and args.ngpus_per_node > 1:
    #     model = torch.nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    # elif torch.cuda.is_available():
    #     model = model.cuda()
    # elif args.ngpus_per_node > 1:
    #     model = torch.nn.DataParallel(model)

    args.transform_func = TransformInput(args)

    # resume from args.weight
    checkpoint = None
    # loc = 'cuda:{}'.format(local_rank if local_rank is not None else 0)
    # if args.weight != 'None':
    #     print("=> loading checkpoint '{}'".format(args.weight))
    #     checkpoint_path = args.weight
    #     checkpoint = torch.load(checkpoint_path, map_location='cpu')
    #     orig_ckt = checkpoint['state_dict']
    #     new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()} 
    #     if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
    #         new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
    #         model.load_state_dict(new_ckt, strict=False)
    #     elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
    #         model.module.load_state_dict(new_ckt, strict=False)
    #     else:
    #         model.load_state_dict(new_ckt, strict=False)
    #     print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))        

    # resume from model_latest
    if not args.not_resume:
        checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> No resume checkpoint found at '{}'".format(checkpoint_path))

    if args.start_epoch < 0:
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] 
        args.start_epoch = max(args.start_epoch, 0)

    if args.eval_only:
        print_str = 'Evaluation ... \n {} Results for checkpoint: {}\n'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), args.weight)
        results_list, hw = evaluate(model, dataloader, local_rank, args, args.dump_vis, huffman_coding=True)
        print_str = f'PSNR for output {hw} for quant {args.quant_str}: '
        for i, (metric_name, best_metric_value, metric_value) in enumerate(zip(args.metric_names, best_metric_list, results_list)):
            best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
            cur_v = RoundTensor(best_metric_value, 2 if 'psnr' in metric_name else 4)
            print_str += f'best_{metric_name}: {cur_v} | '
            best_metric_list[i] = best_metric_value
        if local_rank in [0, None]:
            print(print_str, flush=True)
            with open('{}/eval.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n\n')        
            args.train_time, args.cur_epoch = 0, args.epochs
            Dump2CSV(args, best_metric_list, results_list, [torch.tensor(0)], 'eval.csv')

        return

    # Training
    start = datetime.now()

    psnr_list = []
    cur_avg = 0
    counter = 0
    for sid, sample in enumerate(dataset):
        counter += 1
        for key, value in sample.items():
            for i, v in enumerate(value):
                sample[key][i] = v[None].cuda()
        gt_img = sample["gt_img"][0]

        if args.variant == "b": # or args.variant == "c":
            model_input = sample["downscaled"]
        elif args.variant == "c":
            model_input = sample["laplacians"]
        else:
            model_input = gt_img

        if sid % 10 == 0:
            print("currently working on ", sid, "-th sample.")
        if args.variant == "a":
            model = VariantsA(model=args.model, in_channels=3, out_channels=8, kernel_size=8)
        elif args.variant == "b":
            model = VariantsB(model=args.model, in_channels=3, out_channels=8, kernel_size=8)
        elif args.variant == "c": # or args.variant == "d":
            model = VariantsB(model=args.model, in_channels=3, out_channels=8, kernel_size=8)
        else:
            model = Vanilla(model=args.model, in_channels=3, out_channels=8, kernel_size=8)
        model.cuda()
        if args.dct:
            dct_encoder = SimpleNetEncoder(gt_img.shape[-1], gt_img.shape[-2], drop=0.0, ortho_constraint=False).cuda()
            # Freeze DCT encoder parameters just in case
            for param in dct_encoder.parameters():
                param.requires_grad = False
        # optimizer = Lion(model.parameters(), weight_decay=0.)
        from torch.optim import AdamW

        # ssm_params = []
        # non_ssm_params = []
        
        # if hasattr(model, 'module'):
        #     encoder = model.module.encoder
        #     non_ssm_params.extend(model.module.decoder.parameters())
        # else:
        #     encoder = model.encoder
        #     non_ssm_params.extend(model.decoder.parameters())

        # Group encoder parameters
        # for name, param in encoder.named_parameters():
        #     if any(x in name for x in ['Lambda_re', 'Lambda_im', 'B_', 'log_step']):
        #         # print(f'{name} {param.shape}')
        #         ssm_params.append(param)
        #     else:
        #         non_ssm_params.append(param)

        # param_groups = [
        #     {'params': non_ssm_params, 'weight_decay': 0.0, 'lr': 1e-1},   # Default learning rate for non_ssm_params
        #     {'params': ssm_params, 'weight_decay': 0.0, 'lr': 1e-2},     # Lower learning rate for ssm_params
        # ]
        # decoder_params = sum(p.numel() for name, p in model.named_parameters() if 'decoder' in name)
        # print(f"Number of decoder parameters: {decoder_params:,}")
        # encoder_params = sum(p.numel() for name, p in model.named_parameters() if 'encoder' in name)
        # print(f"Number of encoder parameters: {encoder_params:,}")
        # import pdb; pdb.set_trace()

        # optimizer = AdamW(param_groups)
        optimizer = AdamW(model.parameters(), weight_decay=0.)
        # optimizer = Lion(model.parameters(), weight_decay=0.)
        pred_psnr_list = []
        # epoch_total_start = datetime.now()
        # Save sample image
        # if sid == 0:  # Save first sample only
        #     sample_img = sample.detach().cpu()
        #     # Ensure values are in valid range for saving
        #     sample_img = (sample_img - sample_img.min()) / (sample_img.max() - sample_img.min())
        #     torchvision.utils.save_image(sample_img, 'sample_image.png')
        if args.dct:
            dct_out = torch.log(torch.clamp(dct_encoder(gt_img), min=1.0))
        for epoch in range(args.start_epoch, args.epochs):
            model.train()       
            epoch_start_time = datetime.now()
            # iterate over dataloader
            device = next(model.parameters()).device
            # forward and backward
            cur_epoch = (epoch + 1 / 1) / args.epochs
            lr = adjust_lr(optimizer, cur_epoch, args)
            img_out = model(model_input)
            if args.dct:
                final_loss = loss_fn(img_out, dct_out, "L1")
            else:
                final_loss = loss_fn(img_out, gt_img, args.loss)
            # final_loss += loss_fn(model.encoder(img_out[None]), img_embed)

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            if args.dct:
                mean_abs_error = (img_out - dct_out).abs().mean()
                # pred_psnr_list.append(psnr_fn_single(img_out.detach()[None], dct_out)) 
                pred_psnr_list.append(mean_abs_error.detach())
                pred_psnr = min(pred_psnr_list)
            else:
                pred_psnr_list.append(psnr_fn_single(img_out.detach()[None], gt_img[None])) 
                pred_psnr = max(pred_psnr_list)
                if pred_psnr.isnan():
                    pred_psnr = torch.tensor(0.0, device=pred_psnr.device)
            if epoch == args.epochs - 1:
                metric_string = 'pred_PSNR' if not args.dct else 'pred_L1_loss'
                print_str = '[{}] Rank:{}, Sample ID:{}, Epoch[{}/{}], lr:{:.2e} {}: {}'.format(
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, sid, epoch+1, args.epochs, lr, 
                    metric_string, RoundTensor(pred_psnr, 4))
                print(print_str, flush=True)
                if local_rank in [0, None]:
                    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                        f.write(print_str + '\n')
        # # Plot DCT channels
        # import matplotlib.pyplot as plt
        # # Move tensor to CPU and convert to numpy
        # dct_np = dct_out.detach().cpu().numpy()
        
        # # Create figure with 3 subplots
        # fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # # Plot each channel
        # # Plot original image first
        # im_orig = axes[0].imshow(sample.detach().cpu()[0], cmap='gray')
        # axes[0].set_title('Original Image')
        # plt.colorbar(im_orig, ax=axes[0])
        
        # # Plot DCT channels
        # for i in range(3):
        #     im = axes[i+1].imshow(dct_np[i], cmap='viridis')
        #     axes[i+1].set_title(f'DCT Channel {i}')
        #     plt.colorbar(im, ax=axes[i+1])
        
        # plt.tight_layout()
        # plt.savefig('dct_channels.png')
        # plt.close()
        # epoch_total_end = datetime.now()
        # print(f"Time/epoch: {(epoch_total_end - epoch_total_start).total_seconds():.2f} seconds")
        # import pdb; pdb.set_trace()
        # collect numbers from other gpus
        if args.distributed and args.ngpus_per_node > 1:
            pred_psnr = all_reduce([pred_psnr.to(local_rank)])

        # ADD train_PSNR TO TENSORBOARD
        if local_rank in [0, None]:
            h, w = img_out.shape[-2:]
            writer.add_scalar(f'Train/pred_PSNR_{h}X{w}', pred_psnr, epoch+1)
            writer.add_scalar('Train/lr', lr, epoch+1)
            epoch_end_time = datetime.now()
            print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
                    (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch) ))

        # # evaluation
        # if (epoch + 1) % args.eval_freq == 0 or (args.epochs - epoch) in [1, 3, 5]:
        #     results_list, hw = evaluate(model, full_dataloader, local_rank, args, 
        #         args.dump_vis if epoch == args.epochs - 1 else False, 
        #         True if epoch == args.epochs - 1 else False)            
        #     if local_rank in [0, None]:
        #         # ADD val_PSNR TO TENSORBOARD
        #         print_str = f'Eval at epoch {epoch+1} for {hw}: '
        #         for i, (metric_name, best_metric_value, metric_value) in enumerate(zip(args.metric_names, best_metric_list, results_list)):
        #             best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
        #             if 'psnr' in metric_name:
        #                 writer.add_scalar(f'Val/{metric_name}_{hw}', metric_value.max(), epoch+1)
        #                 writer.add_scalar(f'Val/best_{metric_name}_{hw}', best_metric_value, epoch+1)
        #                 if metric_name == 'pred_seen_psnr':
        #                     psnr_list.append(metric_value.max())
        #                 print_str += f'{metric_name}: {RoundTensor(metric_value, 2)} | '
        #             best_metric_list[i] = best_metric_value
        #         print(print_str, flush=True)
        #         with open('{}/rank0.txt'.format(args.outf), 'a') as f:
        #             f.write(print_str + '\n')

        # state_dict = model.state_dict()
        # save_checkpoint = {
        #     'epoch': epoch+1,
        #     'state_dict': state_dict,
        #     'optimizer': optimizer.state_dict(),   
        # }    
        # if local_rank in [0, None]:
        #     torch.save(save_checkpoint, '{}_{}/model_latest.pth'.format(sid, args.outf))
        #     if (epoch + 1) % args.epochs == 0:
        #         args.cur_epoch = epoch + 1
        #         args.train_time = str(datetime.now() - start)
        #         Dump2CSV(args, best_metric_list, results_list, max(pred_psnr_list), f'epoch{epoch+1}.csv')
        #         torch.save(save_checkpoint, f'{args.outf}/epoch{epoch+1}.pth')
        #         if best_metric_list[0]==results_list[0]:
        #             torch.save(save_checkpoint, f'{args.outf}/model_best.pth')
        # psnr_list.append(max(pred_psnr_list)) 
        
        
        if args.dct:
            cur_mean_abs_error = min(pred_psnr_list)
            cur_avg = (cur_avg*(counter-1) + cur_mean_abs_error) / counter
            cur_avg_rounded = RoundTensor(cur_avg, 4)
            print("current l1 Loss: ", cur_avg_rounded)
        else:
            cur_psnr = max(pred_psnr_list)
            if cur_psnr.isnan():
                cur_psnr = cur_avg
            cur_avg = (cur_avg*(counter-1) + cur_psnr) / counter
            print("current score: ", cur_avg)

    print("final score: ", cur_avg)
    if local_rank in [0, None]:
        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
            f.write(f"final score: {cur_avg}\n")

    if local_rank in [0, None]:
        print(f"Training complete in: {str(datetime.now() - start)}")


# Writing final results in CSV file
def Dump2CSV(args, best_results_list, results_list, psnr_list, filename='results.csv'):
    result_dict = {'Vid':args.vid, 'CurEpoch':args.cur_epoch, 'Time':args.train_time, 
        'FPS':args.fps, 'Split':args.data_split, 'Embed':args.embed, 'Crop': args.crop_list,
        'Resize':args.resize_list, 'Lr_type':args.lr_type, 'LR (E-3)': args.lr*1e3, 'Batch':args.batchSize,
        'Size (M)': f'{round(args.encoder_param, 2)}_{round(args.decoder_param, 2)}_{round(args.total_param, 2)}', 
        'ModelSize': args.modelsize, 'Epoch':args.epochs, 'Loss':args.loss, 'Act':args.act, 'Norm':args.norm,
        'FC':args.fc_hw, 'Reduce':args.reduce, 'ENC_type':args.conv_type[0], 'ENC_strds':args.enc_strd_str, 'KS':args.ks,
        'enc_dim':args.enc_dim, 'DEC':args.conv_type[1], 'DEC_strds':args.dec_strd_str, 'lower_width':args.lower_width,
         'Quant':args.quant_str, 'bits/param':args.bits_per_param, 'bits/param w/ overhead':args.full_bits_per_param, 
        'bits/pixel':args.total_bpp, f'PSNR_list_{args.eval_freq}':','.join([RoundTensor(v, 2) for v in psnr_list]),}
    result_dict.update({f'best_{k}':RoundTensor(v, 4 if 'ssim' in k else 2) for k,v in zip(args.metric_names, best_results_list)})
    result_dict.update({f'{k}':RoundTensor(v, 4 if 'ssim' in k else 2) for k,v in zip(args.metric_names, results_list) if 'pred' in k})
    csv_path = os.path.join(args.outf, filename)
    print(f'results dumped to {csv_path}')
    pd.DataFrame(result_dict,index=[0]).to_csv(csv_path)


@torch.no_grad()
def evaluate(model, full_dataloader, local_rank, args, 
    dump_vis=False, huffman_coding=False):
    img_embed_list = []
    model_list, quant_ckt = quant_model(model, args)
    metric_list = [[] for _ in range(len(args.metric_names))]
    for model_ind, cur_model in enumerate(model_list):
        time_list = []
        cur_model.eval()
        device = next(cur_model.parameters()).device
        if dump_vis:
            visual_dir = f'{args.outf}/visualize_model' + ('_quant' if model_ind else '_orig')
            print(f'Saving predictions to {visual_dir}...')
            if not os.path.isdir(visual_dir):
                os.makedirs(visual_dir)        

        for i, sample in enumerate(full_dataloader):
            img_data, norm_idx, img_idx = data_to_gpu(sample['img'], device), data_to_gpu(sample['norm_idx'], device), data_to_gpu(sample['idx'], device)
            if i > 10 and args.debug:
                break
            img_data, img_gt, inpaint_mask = args.transform_func(img_data)
            cur_input = norm_idx if 'pe' in args.embed else img_data
            img_out, embed_list, dec_time = cur_model(cur_input, dequant_vid_embed[i] if model_ind else None)
            if model_ind == 0:
                img_embed_list.append(embed_list[0])
            
            # collect decoding fps
            time_list.append(dec_time)
            if args.eval_fps:
                time_list.pop()
                for _ in range(100):
                    img_out, embed_list, dec_time = cur_model(cur_input, embed_list[0])
                    time_list.append(dec_time)

            # compute psnr and ms-ssim
            pred_psnr, pred_ssim = psnr_fn_batch([img_out], img_gt), msssim_fn_batch([img_out], img_gt)
            for metric_idx, cur_v in  enumerate([pred_psnr, pred_ssim]):
                for batch_i, cur_img_idx in enumerate(img_idx):
                    metric_idx_start = 2 if cur_img_idx in args.val_ind_list else 0
                    metric_list[metric_idx_start+metric_idx+4*model_ind].append(cur_v[:,batch_i])

            # dump predictions
            if dump_vis:
                for batch_ind, cur_img_idx in enumerate(img_idx):
                    full_ind = i * args.batchSize + batch_ind
                    dump_img_list = [img_data[batch_ind], img_out[batch_ind]]
                    temp_psnr_list = ','.join([str(round(x[batch_ind].item(), 2)) for x in pred_psnr])
                    concat_img = torch.cat(dump_img_list, dim=2)    #img_out[batch_ind], 
                    save_image(concat_img, f'{visual_dir}/pred_{full_ind:04d}_{temp_psnr_list}.png')

            # print eval results and add to log txt
            if i % args.print_freq == 0 or i == len(full_dataloader) - 1:
                avg_time = sum(time_list) / len(time_list)
                fps = args.batchSize / avg_time
                print_str = '[{}] Rank:{}, Eval at Step [{}/{}] , FPS {}, '.format(
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, i+1, len(full_dataloader), round(fps, 1))
                metric_name = ('quant' if model_ind else 'pred') + '_seen_psnr'
                for v_name, v_list in zip(args.metric_names, metric_list):
                    if metric_name in v_name:
                        cur_value = torch.stack(v_list, dim=-1).mean(-1) if len(v_list) else torch.zeros(1)
                        print_str += f'{v_name}: {RoundTensor(cur_value, 2)} | '
                if local_rank in [0, None]:
                    print(print_str, flush=True)
                    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                        f.write(print_str + '\n')
        
        # embedding quantization
        if model_ind == 0:
            vid_embed = torch.cat(img_embed_list, 0) 
            quant_embed, dequant_emved = quant_tensor(vid_embed, args.quant_embed_bit)
            dequant_vid_embed = dequant_emved.split(args.batchSize, dim=0)

        # Collect results from 
        results_list = [torch.stack(v_list, dim=1).mean(1).cpu() if len(v_list) else torch.zeros(1) for v_list in metric_list]
        args.fps = fps
        h,w = img_data.shape[-2:]
        cur_model.train()
        if args.distributed and args.ngpus_per_node > 1:
            for cur_v in results_list:
                cur_v = all_reduce([cur_v.to(local_rank)])

        # Dump predictions and concat into videos
        if dump_vis and args.dump_videos:
            gif_file = os.path.join(args.outf, 'gt_pred' + ('_quant.gif' if model_ind else '.gif'))
            with imageio.get_writer(gif_file, mode='I') as writer:
                for filename in sorted(os.listdir(visual_dir)):
                    image = imageio.v2.imread(os.path.join(visual_dir, filename))
                    writer.append_data(image)
            if not args.dump_images:
                shutil.rmtree(visual_dir)
            # optimize(gif_file)
        
    # dump quantized checkpoint, and decoder
    if local_rank in [0, None] and quant_ckt != None:
        quant_vid = {'embed': quant_embed, 'model': quant_ckt}
        torch.save(quant_vid, f'{args.outf}/quant_vid.pth')
        # torch.jit.save(torch.jit.trace(HNeRVDecoder(model), (vid_embed[:2])), f'{args.outf}/img_decoder.pth')
        # huffman coding
        if huffman_coding:
            quant_v_list = quant_embed['quant'].flatten().tolist()
            tmin_scale_len = quant_embed['min'].nelement() + quant_embed['scale'].nelement()
            for k, layer_wt in quant_ckt.items():
                quant_v_list.extend(layer_wt['quant'].flatten().tolist())
                tmin_scale_len += layer_wt['min'].nelement() + layer_wt['scale'].nelement()

            # get the element name and its frequency
            unique, counts = np.unique(quant_v_list, return_counts=True)
            num_freq = dict(zip(unique, counts))

            # generating HuffmanCoding table
            codec = HuffmanCodec.from_data(quant_v_list)
            sym_bit_dict = {}
            for k, v in codec.get_code_table().items():
                sym_bit_dict[k] = v[0]

            # total bits for quantized embed + model weights
            total_bits = 0
            for num, freq in num_freq.items():
                total_bits += freq * sym_bit_dict[num]
            args.bits_per_param = total_bits / len(quant_v_list)
            
            # including the overhead for min and scale storage, 
            total_bits += tmin_scale_len * 16               #(16bits for float16)
            args.full_bits_per_param = total_bits / len(quant_v_list)

            # bits per pixel
            args.total_bpp = total_bits / args.final_size / args.full_data_length
            print(f'After quantization and encoding: \n bits per parameter: {round(args.full_bits_per_param, 2)}, bits per pixel: {round(args.total_bpp, 4)}')
    # import pdb; pdb.set_trace; from IPython import embed; embed()     

    return results_list, (h,w)


def quant_model(model, args):
    model_list = [deepcopy(model)]
    if args.quant_model_bit == -1:
        return model_list, None
    else:
        cur_model = deepcopy(model)
        quant_ckt, cur_ckt = [cur_model.state_dict() for _ in range(2)]
        encoder_k_list = []
        for k,v in cur_ckt.items():
            if 'encoder' in k:
                encoder_k_list.append(k)
            else:
                quant_v, new_v = quant_tensor(v, args.quant_model_bit)
                quant_ckt[k] = quant_v
                cur_ckt[k] = new_v
        for encoder_k in encoder_k_list:
            del quant_ckt[encoder_k]
        cur_model.load_state_dict(cur_ckt)
        model_list.append(cur_model)
        
        return model_list, quant_ckt


if __name__ == '__main__':
    main()
