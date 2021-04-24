from model import Generator, Discriminator
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


def build_models(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    netG_A2B = Generator(args.input_nc, args.output_nc)
    netG_B2A = Generator(args.input_nc, args.output_nc)
    netD_A = Discriminator(args.input_nc)
    netD_B = Discriminator(args.output_nc)

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    print('Weights Initialized')

    netG_A2B.to(device)
    netG_B2A.to(device)
    netD_A.to(device) 
    netD_B.to(device) 

    print(f'Transferred to {device}')

    return netG_A2B, netG_B2A, netD_A, netD_B


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)



def train_generators(photo, monet_img, netG_A2B, netG_B2A, criterion_identity, criterion_GAN, criterion_cycle, optimizer_G)

    ###### Generators A2B and B2A ######
    optimizer_G.zero_grad()

    # Identity loss
    # G_A2B(B) should equal B if real B is fed
    same_B = netG_A2B(real_B)
    loss_identity_B = criterion_identity(same_B, real_B)*5.0
    # G_B2A(A) should equal A if real A is fed
    same_A = netG_B2A(real_A)
    loss_identity_A = criterion_identity(same_A, real_A)*5.0

    # GAN loss
    fake_B = netG_A2B(real_A)
    pred_fake = netD_B(fake_B)
    loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

    fake_A = netG_B2A(real_B)
    pred_fake = netD_A(fake_A)
    loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

    # Cycle loss
    recovered_A = netG_B2A(fake_B)
    loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

    recovered_B = netG_A2B(fake_A)
    loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

    # Total loss
    loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
    loss_G.backward(retain_graph = True)

    optimizer_G.step()

    gen_losses = (loss_G, loss_identity_A + loss_identity_B, loss_GAN_A2B + loss_GAN_B2A, loss_cycle_ABA + loss_cycle_BAB)

    return gen_losses, fake_A, fake_B


def train_discriminators(photo, monet_img, fake_A, fake_B, netD_A, netD_B, criterion_GAN, optimizer_D_A, optimizer_D_B):


    ###### Discriminator A ######
    optimizer_D_A.zero_grad()

    # Real loss
    pred_real = netD_A(real_A)
    loss_D_real = criterion_GAN(pred_real, target_real * 0.9)

    # Fake loss
    pred_fake = netD_A(fake_A)
    loss_D_fake = criterion_GAN(pred_fake, target_fake)

    # Total loss
    loss_D_A = (loss_D_real + loss_D_fake)*0.5
    loss_D_A.backward()

    
    ###################################

    ###### Discriminator B ######
    optimizer_D_B.zero_grad()

    # Real loss
    pred_real = netD_B(real_B)
    loss_D_real = criterion_GAN(pred_real, target_real * 0.9)
    
    # Fake loss
    pred_fake = netD_B(fake_B)
    loss_D_fake = criterion_GAN(pred_fake, target_fake)

    # Total loss
    loss_D_B = (loss_D_real + loss_D_fake)*0.5
    loss_D_B.backward()

    optimizer_D_A.step()
    optimizer_D_B.step()

    dis_losses = loss_D_A + loss_D_B

    return dis_losses