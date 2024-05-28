import os
import builtins
import glob
import argparse
import torch
import numpy as np 
import random
import torch.distributed as dist
from loader import get_trn_dev_loader, get_tst_loader
from trainer import Trainer
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer

torch.backends.cudnn.benchmark = True

def distributed_launch(args):
    """
    Initialize distributed training parameters based on the environment variables and 
    the method used to launch the training job (torchrun, srun and torch.distributed.launch).
    Copy this method to distributed launch. No changes required.
    """
    
    if "WORLD_SIZE" in os.environ: # torchrun (option 2)
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_NTASKS" in os.environ: # srun launch 
        args.world_size = int(os.environ['SLURM_NTASKS'])
    args.distributed = args.world_size > 1

    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch (option 3 and option 2)
            args.global_rank = int(os.environ["RANK"])
            args.gpu_id = args.global_rank % torch.cuda.device_count()
            args.local_rank = args.gpu_id
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler srun (option 1)
            args.global_rank = int(os.environ['SLURM_PROCID'])
            args.gpu_id = args.global_rank % torch.cuda.device_count()
            args.local_rank = args.gpu_id
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.global_rank)

    
def main(args):
    
    distributed_launch(args)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    print("\n[Step 0] Check GPUs and process assigment\n")
    print("_____________________________________________")
    print("This is process number ", args.global_rank, " set to GPU device number (local rank:", args.local_rank , "== local gpu:", args.gpu_id, ")")
    print("_____________________________________________")
    dist.barrier()

    # suppress printing if not on master gpu
    #if args.global_rank != 0:
    #    def print_pass(*args):
    #        pass
    #    builtins.print = print_pass


    print("\n[Step 1] Loading dataset\n")
    loaders = get_trn_dev_loader(
        dset=load_dataset("imdb", split="train"),
        tok=tokenizer,
        batch_size=args.batch_size,
        workers=args.workers,
        rank=args.global_rank,
        distributed=args.distributed,
    )

    print("\n[Step 2] Loading model\n")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Parameters] {num_params}")
    
    
    ### resume training from checkpoint (note that one checkpoint per global_rank) ###
    if args.resume:
        pass
        
    # training phase
    print("\n[Step 3] Training is starting\n")

    trainer = Trainer(args, tokenizer, loaders, model)
    best_result = trainer.fit(num_ckpt=1)


    # testing phase
    if args.global_rank in [-1, 0]:
        version = best_result["version"]
        state_dict = torch.load(
            glob.glob(
                os.path.join(args.ckpt_path, f"version-{version}/best_model_*.pt")
            )[0]
        )
        test_loader = get_tst_loader(
            dset=load_dataset("imdb", split="test"),
            tok=tokenizer,
            batch_size=args.batch_size,
            workers=args.workers,
            distributed=args.distributed,
        )
        
        
        test_result = trainer.test(test_loader, state_dict)

    # save results to
    print("Save results to file.")	

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=123, type=int, help='seed')
    parser.add_argument('--model', default='resnet18', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--gpu-id', default=None, type=int)
    parser.add_argument('--start_epoch_from_checkpoint', default=0, type=int, 
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--global-rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local-rank', default=-1, type=int, 
                        help='local rank for distributed training')
    parser.add_argument('--mode', default='train', type=str, 
                        help='(train/test)')
    parser.add_argument('--workers', default=4, type=int, 
                        help='workers for dataloader')
    parser.add_argument('--max-grad-norm', default=1.0, type=float, 
                        help='grad norm')
    parser.add_argument('--gradient-accumulation-step', default=1, type=int, 
                        help='number of gradient accumulations')
    parser.add_argument('--distributed', default=True, type=bool, 
                        help='(True/False)')
    parser.add_argument('--resume', default=False, type=bool, 
                        help='from checkpoint (True/False)')
    parser.add_argument('--ckpt-path', default="./chkpt/", type=str, 
                        help='Path to checkpoint save folder')
    parser.add_argument("--result-path", type=str, default="src/results.csv")
    parser.add_argument('--eval-ratio', default=1, type=int, 
                        help='epochs required for evaluation')
    parser.add_argument('--log-step', default=200, type=int)
    parser.add_argument("--weight-decay", type=float, default=4e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--amp", action="store_true", default=False, help="PyTorch(>=1.6.x) AMP"
    )
                                                
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    print(torch.cuda.is_available())
    args = parse_args()
    main(args)
