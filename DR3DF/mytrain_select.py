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
import torch.nn as nn
from My_Model_select import Score
from Model_util import padding_image
from make import getTxt
from perceptual import LossNetwork
from pytorch_msssim import msssim
from test_dataset import dehaze_test_dataset
from train_dataset import dehaze_train_dataset
from utils_test import to_psnr, to_ssim_skimage

ssl._create_default_https_context = ssl._create_unverified_context


def Gumbel_Softmax_delrandom(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='Dehaze Network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=4, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=2000, type=int)
parser.add_argument("--type", default=0, type=int, help="choose a type 012345")

# parser.add_argument('--train_dir', type=str, default='')
parser.add_argument('--train_dir', type=str, default='')
parser.add_argument('--train_name', type=str, default='hazy,clean')
parser.add_argument('--test_dir', type=str, default='')
parser.add_argument('--test_name', type=str, default='hazy,clean')

parser.add_argument('--model_save_dir', type=str, default='./output_result_select')
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--gpus', default='4', type=str)
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
# parser.add_argument('--mode', type=str, default='train', help='train or test')

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

# D3D = Base_Model(3, 3, mode = args.mode)
score = Score(3, 3)
score = score.to(device)

# --- Build optimizer --- #
G_optimizer = torch.optim.Adam(score.parameters(), lr=0.0001)
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[5000, 7000, 8000], gamma=0.5)

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


writer = SummaryWriter(os.path.join(args.model_save_dir, 'tensorboard'))

# --- Load the network weight --- #
if args.restart:
    pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
    num = sorted([int(i.split('.')[0].split('_')[1]) for i in pkl_list])[-1]
    name = [i for i in pkl_list if 'epoch_' + str(num) + '_' in i][0]
    score.load_state_dict(
        torch.load(os.path.join(args.model_save_dir, name),
                   map_location="cuda:{}".format(device_ids[0])))
    print('--- {} epoch weight loaded ---'.format(num))
    start_epoch = int(num) + 1
elif args.num != '9999999':
    pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
    name = [i for i in pkl_list if 'epoch_' + str(args.num) + '_' in i][0]
    score.load_state_dict(
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
    score.train()
    with tqdm(total=len(train_loader)) as t:
        for (hazy, clean) in train_loader:
            iteration += 1
            hazy = hazy.to(device)
            clean = clean.to(device)

            fea_map, score_u, score_f, score_fu, out_u, out_f, out_fu = score(hazy)
            smooth_l1_U = F.smooth_l1_loss(clean, out_u, reduction='none').mean(dim=[1, 2, 3])
            smooth_l1_F = F.smooth_l1_loss(clean, out_f, reduction='none').mean(dim=[1, 2, 3])
            smooth_l1_out = F.smooth_l1_loss(clean, out_fu, reduction='none').mean(dim=[1, 2, 3])

            score_map = torch.cat((score_u, score_f, score_fu), dim=1)  # [3]
            score_idx = Gumbel_Softmax_delrandom(score_map, hard=True, tau=1, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out_map = fea_map * score_idx

            out = score.D3D.conv(out_map.sum(1))

            clean_label = torch.cat((smooth_l1_U.unsqueeze(1), smooth_l1_F.unsqueeze(1), smooth_l1_out.unsqueeze(1)),dim=1)

            # print(clean_label)
            ranked_label = torch.zeros_like(clean_label)
            max_indices = torch.argmax(clean_label, dim=1)
            ranked_label[range(clean_label.size(0)), max_indices] = -1

            temp_label = clean_label.clone()
            temp_label[range(clean_label.size(0)), max_indices] = float('-inf')
            second_max_indices = torch.argmax(temp_label, dim=1)
            ranked_label[range(clean_label.size(0)), second_max_indices] = 0

            min_indices = torch.argmin(clean_label, dim=1)
            ranked_label[range(clean_label.size(0)), min_indices] = 1

            ranked_label = ranked_label.view(score_map.shape)

            #print(score_map)
            #print(score_idx)
            total_loss = torch.mean(torch.abs(score_map - ranked_label))
            total_loss.backward()
            G_optimizer.step()
            score.zero_grad()

            writer.add_scalars('training_score', {'training total loss': total_loss.item()}, iteration)

            t.set_description("===> Epoch[{}] :  total_loss: {:.8f} ".format(epoch, total_loss.item(),time.time() - start_time))
            t.update(1)

    if args.seps:
        torch.save(score.state_dict(),
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

    if epoch % 1 == 0:

        with torch.no_grad():
            i = 0
            psnr_list = []
            ssim_list = []
            score.eval()
            for (hazy, clean, _) in tqdm(test_loader):
                hazy = hazy.to(device)
                clean = clean.to(device)

                h, w = hazy.shape[2], hazy.shape[3]
                max_h = int(math.ceil(h / 4)) * 4
                max_w = int(math.ceil(w / 4)) * 4
                hazy, ori_left, ori_right, ori_top, ori_down = padding_image(hazy, max_h, max_w)

                fea_map, score_u, score_f, score_fu, out_u, out_f, out_fu = score(hazy)
                score_map = torch.cat((score_u, score_f, score_fu), dim=1)  # [3]


                score_idx = Gumbel_Softmax_delrandom(score_map, hard=True).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                print(score_idx)

                out_map = fea_map * score_idx
                out = score.D3D.conv(out_map.sum(1))
                out = out.data[:, :, ori_top:ori_down, ori_left:ori_right]
                psnr_list.extend(to_psnr(out, clean))
                ssim_list.extend(to_ssim_skimage(out, clean))

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
            frame_debug = torch.cat((out, clean), dim=0)
            writer.add_images('my_image_batch', frame_debug, epoch)
            writer.add_scalars('testing', {'testing psnr': avr_psnr,
                                           'testing ssim': avr_ssim
                                           }, epoch)
            if best_epoch_psnr == epoch or best_epoch_ssim == epoch:
                torch.save(score.state_dict(),
                           os.path.join(args.model_save_dir,
                                        'epoch_' + str(epoch) + '_' + str(round(avr_psnr, 2)) + '_' + str(
                                            round(avr_ssim, 3)) + '_' + str(tag) + '.pkl'))
os.remove(os.path.join(args.train_dir, 'train.txt'))
os.remove(os.path.join(args.test_dir, 'test.txt'))
writer.close()
