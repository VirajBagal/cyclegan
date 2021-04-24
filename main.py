import argparse
import itertools
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import IMGDataset
from utils import build_models, LambdaLR, train_generators, train_discriminators
import wandb



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--photos_path", required = True, help='path to photos')
    parser.add_argument("--paintings_path", required = True, help='path to paintings')
    parser.add_argument("--run_name", type=str, required = True)
    parser.add_argument("--input_nc", default=3, type=int, help="number of input channels") 
    parser.add_argument("--output_nc", default=3, type=int, help="number of output channels") 
    parser.add_argument("--epoch", default=0, type=int, help="offset epochs") 
    parser.add_argument("--n_epochs", default=30, type=int, help="total epochs") 
    parser.add_argument("--decay_epoch", default=30, type=int, help="deacy of lr starts") 
    parser.add_argument("--batch_size", default=16, type=int, help="batch_size")
    parser.add_argument("--lr", default=1e-4, type=float, help='learning_rate')
    parser.add_argument("--img_size", default=224, type=int)

    args = parser.parse_args()

    wandb.init(project = 'cyclegan', name = args.run_name)

    if not os.path.exists(f'checkpoints/{args.run_name}'):
        os.makedirs(f'checkpoints/{args.run_name}')

    tfm =  transforms.Compose([transforms.ToTensor()])
    dataset = IMGDataset(args, tfm)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4, pin_memory=True)

    netG_A2B, netG_B2A, netD_A, netD_B = build_models(args)

    # Losses
    criterion_GAN = torch.nn.BCEWithLogitsLoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                    lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_real = torch.ones(args.batch_size, dtype=torch.float).unsqueeze(1).to(device) 
    target_fake = torch.ones(args.batch_size, dtype=torch.float).unsqueeze(1).to(device) 

    wandb_step = 0
    log_image_step = 50
    loader_len = len(dataloader)

    ###### Training ######
    for epoch in range(args.epoch, args.n_epochs):
        for i, (photo, monet_img) in enumerate(tqdm(dataloader, total = loader_len)):

            real_A = photo.to(device)
            real_B = monet_img.to(device)
            
            gen_losses, fake_A, fake_B = train_generators(real_A, real_B, netG_A2B, netG_B2A, criterion_identity, criterion_GAN, criterion_cycle, optimizer_G)
            dis_losses = train_discriminators(real_A, real_B, fake_A, fake_B, netD_A, netD_B, criterion_GAN, optimizer_D_A, optimizer_D_B)

            loss_G, loss_G_identity, loss_G_GAN, loss_G_cycle = gen_losses

            wandb_step = loader_len*epoch + i
            wandb.log({'loss_G': loss_G, 'loss_G_identity': loss_G_identity, 'loss_G_GAN': loss_G_GAN,
                        'loss_G_cycle': loss_G_cycle, 'loss_D': dis_losses}, step = wandb_step)
            
            if wandb_step % log_image_step == 0:
        
                wandb.log({'exp03': [wandb.Image(real_A.cpu().detach().numpy()[0].transpose(1,2,0), caption='real_A'), 
                                    wandb.Image(real_B.cpu().detach().numpy()[0].transpose(1,2,0), caption='real_B'),
                                    wandb.Image(fake_A.cpu().detach().numpy()[0].transpose(1,2,0), caption = 'fake_A'),
                                    wandb.Image(fake_B.cpu().detach().numpy()[0].transpose(1,2,0), caption = 'fake_B')]}, step = wandb_step)



        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()







    

