import argparse
import math
import os
import ssl
import time

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from torchvision.utils import save_image
from tqdm import tqdm

from My_Model_dehazing import Base_Model,Discriminator
from Model_util import padding_image
from make import getTxt
from perceptual import LossNetwork
from pytorch_msssim import msssim
from test_dataset import dehaze_test_dataset
from train_dataset import dehaze_train_dataset
from utils_test import to_psnr, to_ssim_skimage

ssl._create_default_https_context = ssl._create_unverified_context


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False

# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='Dehaze Network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=2, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=2000, type=int)
parser.add_argument("--type", default=0, type=int, help="choose a type 012345")

# parser.add_argument('--train_dir', type=str, default='')
parser.add_argument('--train_dir', type=str, default='')
parser.add_argument('--train_name', type=str, default='hazy,clean')
parser.add_argument('--test_dir', type=str, default='')
parser.add_argument('--test_name', type=str, default='hazy,clean')

parser.add_argument('--model_save_dir', type=str, default='./output_result_dehazing')
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--gpus', default='5', type=str)
# --- Parse hyper-parameters test --- #
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
parser.add_argument('--vgg_model', default='', type=str, help='load trained model or not')
parser.add_argument('--use_bn', action='store_true', help='if bs>8 please use bn')
parser.add_argument('--restart', action='store_true', help='')
parser.add_argument('--num', type=str, default='9999999', help='')
parser.add_argument('--sep', type=int, default='5', help='')
parser.add_argument('--save_psnr', action='store_true', help='')
parser.add_argument('--seps', action='store_true', help='')
#parser.add_argument('--mode', type=str, default='train', help='train or test')

print('+++++++++++++++++++++++++++++++ Train set ++++++++++++++++++++++++++++++++++++++++')

args = parser.parse_args()
# --- train --- #
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch = args.train_epoch
start_epoch = 0
sep = args.sep

if args.type == 1:
    args.train_dir = 'path your train dataset'
    args.train_name = 'hazy,clean'
    args.test_dir = "path your test dataset"
    args.test_name = 'hazy,clean'
    args.out_pic = './output_pic/thin'
    tag = 'thin'



print('We are training datasets: ', tag)

getTxt(args.train_dir, args.train_name, args.test_dir, args.test_name)

predict_result = args.predict_result
test_batch_size = args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
# output_dir = os.path.join(args.model_save_dir, 'output_result')

# --- Gpu device --- #
device_ids = [int(i) for i in list(filter(str.isdigit, args.gpus))]

print('use gpus ->', args.gpus)
device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
if args.use_bn:
    print('we are using BatchNorm')
else:
    print('we are using InstanceNorm')


#D3D = Base_Model(3, 3, mode = args.mode)
D3D = Base_Model(3, 3)

print('D3D parameters:', sum(param.numel() for param in D3D.parameters()))
DNet = Discriminator()
print('Discriminator parameters:', sum(param.numel() for param in DNet.parameters()))

# --- Build optimizer --- #
G_optimizer = torch.optim.Adam(D3D.parameters(), lr=0.0001)


scheduler_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[5000, 7000, 8000], gamma=0.5)
D_optim = torch.optim.Adam(DNet.parameters(), lr=0.0001)
scheduler_D = torch.optim.lr_scheduler.MultiStepLR(D_optim, milestones=[5000, 7000, 8000], gamma=0.5)

# --- Load training data --- #
dataset = dehaze_train_dataset(args.train_dir, args.train_name, tag)
print('trainDataset len: ', len(dataset))
train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, drop_last=True,
                          num_workers=4)
# --- Load testing data --- #

test_dataset = dehaze_test_dataset(args.test_dir, args.test_name, tag)
print('testDataset len: ', len(test_dataset))
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0,
                         pin_memory=True)

# val_dataset = dehaze_val_dataset(val_dataset)
# val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

# --- Multi-GPU --- #
D3D = D3D.to(device)
DNet = DNet.to(device)
writer = SummaryWriter(os.path.join(args.model_save_dir, 'tensorboard'))

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True)
# vgg_model.load_state_dict(torch.load(os.path.join(args.vgg_model , 'vgg16.pth')))
vgg_model = vgg_model.features[:16].to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

loss_network = LossNetwork(vgg_model)
loss_network.eval()
msssim_loss = msssim


# --- Load the network weight --- #
if args.restart:
    pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
    num = sorted([int(i.split('.')[0].split('_')[1]) for i in pkl_list])[-1]
    name = [i for i in pkl_list if 'epoch_' + str(num) + '_' in i][0]
    D3D.load_state_dict(
        torch.load(os.path.join(args.model_save_dir, name),
                   map_location="cuda:{}".format(device_ids[0])))
    print('--- {} epoch weight loaded ---'.format(num))
    start_epoch = int(num) + 1
elif args.num != '9999999':
    pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
    name = [i for i in pkl_list if 'epoch_' + str(args.num) + '_' in i][0]
    D3D.load_state_dict(
        torch.load(os.path.join(args.model_save_dir, name),
                   map_location="cuda:{}".format(device_ids[0])))
    print('--- {} epoch weight loaded ---'.format(args.num))
    start_epoch = int(args.num) + 1
else:
    print('--- no weight loaded ---')

iteration = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
pl = []
sl = []
best_psnr = 0
best_psnr_ssim = 0
best_ssim = 0
best_ssim_psnr = 0

start_time = time.time()
for epoch in range(start_epoch, train_epoch):
    print('++++++++++++++++++++++++ {} Datasets +++++++ {} epoch ++++++++++++++++++++++++'.format(tag, epoch))
    scheduler_G.step()
    D3D.train()
    with tqdm(total=len(train_loader)) as t:
        for (hazy, clean) in train_loader:
            iteration += 1
            hazy = hazy.to(device)
            clean = clean.to(device)
            out_u, out_f, out_fu, x_u, x_f, x_fu = D3D(hazy)
            DNet.zero_grad()

            real_out = DNet(clean).mean()
            img_out_u = DNet(out_u).mean()
            img_out_f = DNet(out_f).mean()
            img_out_fu = DNet(out_fu).mean()
            D_loss1 = 1 - real_out + img_out_u
            D_loss2 = 1 - real_out + img_out_f
            D_loss3 = 1 - real_out + img_out_fu
            D_loss = D_loss1+D_loss2+D_loss3
            # no more forward
            D_loss.backward(retain_graph=True)
            D3D.zero_grad()

            u_adversarial_loss = torch.mean(1 - out_u)
            u_smooth_loss_l1 = F.smooth_l1_loss(out_u, clean)
            u_perceptual_loss = loss_network(out_u, clean)
            u_msssim_loss_ = -msssim_loss(out_u, clean, normalize=True)

            f_adversarial_loss = torch.mean(1 - out_f)
            f_smooth_loss_l1 = F.smooth_l1_loss(out_f, clean)
            f_perceptual_loss = loss_network(out_f, clean)
            f_msssim_loss_ = -msssim_loss(out_f, clean, normalize=True)

            fu_adversarial_loss = torch.mean(1 - out_fu)
            fu_smooth_loss_l1 = F.smooth_l1_loss(out_fu, clean)
            fu_perceptual_loss = loss_network(out_fu, clean)
            fu_msssim_loss_ = -msssim_loss(out_fu, clean, normalize=True)

            smooth_loss_l1 = u_smooth_loss_l1+f_smooth_loss_l1+fu_smooth_loss_l1
            perceptual_loss = u_perceptual_loss + f_perceptual_loss +fu_perceptual_loss
            adversarial_loss = u_adversarial_loss + f_adversarial_loss + fu_adversarial_loss
            msssim_loss_ = u_msssim_loss_ + f_msssim_loss_ + fu_msssim_loss_

            total_loss = smooth_loss_l1 + 0.01 * perceptual_loss + 0.0005 * adversarial_loss+ 0.5 * msssim_loss_

            total_loss.backward()
            D_optim.step()
            G_optimizer.step()
            writer.add_scalars('training', {'training total loss': total_loss.item()
                                            }, iteration)
            writer.add_scalars('training_img', {'img loss_l1': smooth_loss_l1.item(),
                                                'perceptual': perceptual_loss.item(),
                                                'msssim': msssim_loss_.item()

                                                }, iteration)
            writer.add_scalars('GAN_training', {
                'd_loss': D_loss.item(),
                'd_score': real_out.item(),
                # 'g_score': fake_out.item()
            }, iteration)
            t.set_description(
                "===> Epoch[{}] :  total_loss: {:.2f} ".format(
                    epoch, total_loss.item(),
                    time.time() - start_time))
            t.update(1)






    if args.seps:
        torch.save(D3D.state_dict(),
                   os.path.join(args.model_save_dir,
                                'epoch_' + str(epoch) + '_' + '.pkl'))
        continue

    if tag in []:
        if epoch >= 30:
            sep = 1
    elif tag in ['thin', 'thick', 'moderation', 'RICE1', 'RICE2', 'RSID', 'NID', 'moderation_pt', 'RICE1_ALL',
                 'RICE2_ALL', 'NID_dense', 'NID_thin']:
        if epoch >= 100:
            sep = 1
    else:
        if epoch >= 500:
            sep = 1

    if epoch >= 0:

        with torch.no_grad():
            i = 0
            psnr_list = []
            ssim_list = []
            D3D.eval()
            for (hazy, clean, _) in tqdm(test_loader):
                hazy = hazy.to(device)
                clean = clean.to(device)

                h, w = hazy.shape[2], hazy.shape[3]
                max_h = int(math.ceil(h / 4)) * 4
                max_w = int(math.ceil(w / 4)) * 4
                hazy, ori_left, ori_right, ori_top, ori_down = padding_image(hazy, max_h, max_w)

                frame_out, out_f, out_fu, x_u, x_f, x_fu = D3D(hazy)
                print("u",to_psnr(frame_out, clean))
                print("f",to_psnr(out_f, clean))
                print("fu",to_psnr(out_fu, clean))
                # if i % 200 == 0:
                #    save_image(frame_out, os.path.join(args.out_pic, str(epoch) + '_' + str(i) + '_' + '.png'))
                # i = i + 1

                frame_out = frame_out.data[:, :, ori_top:ori_down, ori_left:ori_right]

                psnr_list.extend(to_psnr(frame_out, clean))
                ssim_list.extend(to_ssim_skimage(frame_out, clean))

            avr_psnr = sum(psnr_list) / len(psnr_list)
            avr_ssim = sum(ssim_list) / len(ssim_list)
            pl.append(avr_psnr)
            sl.append(avr_ssim)
            if avr_psnr >= max(pl):
                best_epoch_psnr = epoch
                best_psnr = avr_psnr
                best_psnr_ssim = avr_ssim
            if avr_ssim >= max(sl):
                best_epoch_ssim = epoch
                best_ssim = avr_ssim
                best_ssim_psnr = avr_psnr

            print(epoch, 'dehazed', avr_psnr, avr_ssim)
            if best_epoch_psnr == best_epoch_ssim:
                print('best epoch is {}, psnr: {}, ssim: {}'.format(best_epoch_psnr, best_psnr, best_ssim))
            else:
                print('best psnr epoch is {}: PSNR: {}, ssim: {}'.format(best_epoch_psnr, best_psnr, best_psnr_ssim))
                print('best ssim epoch is {}: psnr: {}, SSIM: {}'.format(best_epoch_ssim, best_ssim_psnr, best_ssim))
            print()
            frame_debug = torch.cat((frame_out, clean), dim=0)
            writer.add_images('my_image_batch', frame_debug, epoch)
            writer.add_scalars('testing', {'testing psnr': avr_psnr,
                                           'testing ssim': avr_ssim
                                           }, epoch)
            if best_epoch_psnr == epoch or best_epoch_ssim == epoch:
                torch.save(D3D.state_dict(),
                           os.path.join(args.model_save_dir,
                                        'epoch_' + str(epoch) + '_' + str(round(avr_psnr, 2)) + '_' + str(
                                            round(avr_ssim, 3)) + '_' + str(tag) + '.pkl'))
os.remove(os.path.join(args.train_dir, 'train.txt'))
os.remove(os.path.join(args.test_dir, 'test.txt'))
writer.close()


