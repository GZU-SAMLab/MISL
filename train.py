import argparse
import json
import os
import warnings

import numpy as np
import numpy.random as random
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms

from datasets import TextDataset, prepare_data
from loss import calc_G_loss, calc_D_loss, words_loss, sent_loss
from model import InpaintNet, PatchDiscriminator, RNN_ENCODER, CNN_ENCODER

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='birds')  # birds flowers celeba15000 coco
parser.add_argument('--root', type=str, default='../../../datasets')
parser.add_argument('--save_dir', type=str, default='./model')
parser.add_argument('--training_image', type=str, default='./training')
parser.add_argument('--CAPTIONS_PER_IMAGE', type=int, default=10)
parser.add_argument('--WORDS_NUM', type=int, default=16)
parser.add_argument('--lr', type=float, default=2e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--threshold", type=float, default=0.8, help="RN_L threshold")
parser.add_argument('--max_epoch', type=int, default=400)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--vis_interval', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--Rec', type=float, default=0.01)
parser.add_argument('--TGA', type=float, default=1)
parser.add_argument('--M', type=float, default=0.01)
parser.add_argument('--TIM', type=float, default=0.002)
parser.add_argument('--G', type=float, default=0.004)
parser.add_argument('--LD', type=int, default=2)
parser.add_argument('--mask_start_num', type=int, default=0)
parser.add_argument('--mask_end_num', type=int, default=15000)
parser.add_argument('--mask_file', type=str, default='../../../datasets/mask_latest/train')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

print('dataset:', args.dataset)
print('root:', args.root)
print('save_dir:', args.save_dir)
print('training_image:', args.training_image)
print('CAPTIONS_PER_IMAGE:', args.CAPTIONS_PER_IMAGE)
print('WORDS_NUM:', args.WORDS_NUM)
print('max_epoch:', args.max_epoch)
print('start_epoch:', args.start_epoch)
print('batch_size:', args.batch_size)
print('save_model_interval:', args.save_model_interval)
print('vis_interval:', args.vis_interval)
print('log_interval:', args.log_interval)
print('resume:', args.resume)
print('Rec:', args.Rec)
print('TGA:', args.TGA)
print('M:', args.M)
print('TIM:', args.TIM)
print('G:', args.G)
print('LD:', args.LD)
print('mask_start_num:', args.mask_start_num)
print('mask_end_num:', args.mask_end_num)
print('mask_file:', args.mask_file)

save_dir = args.save_dir + '/' + args.dataset + '_mask-' + str(args.mask_start_num) + '-' + str(args.mask_end_num) + '_Rec-' + str(args.Rec) + '_TGA-' + str(args.TGA) + '_M-' + str(args.M) + '_TIM-' + str(args.TIM) + '_G-' + str(args.G) + '_LD-' + str(args.LD)

if use_cuda:
    torch.backends.cudnn.benchmark = True
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

training_image = args.training_image + '/' + args.dataset + '_mask-' + str(args.mask_start_num) + '-' + str(args.mask_end_num) + '_Rec-' + str(args.Rec) + '_TGA-' + str(args.TGA) + '_M-' + str(args.M) + '_TIM-' + str(args.TIM) + '_G-' + str(args.G) + '_LD-' + str(args.LD)

if not os.path.exists(training_image):
    os.makedirs(training_image)

size = (args.image_size, args.image_size)
train_tf = transforms.Compose([
    transforms.Resize(size),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomGrayscale(),
])

dataset_train = TextDataset(args.root + '/' + args.dataset, base_size=args.image_size,
                            CAPTIONS_PER_IMAGE=args.CAPTIONS_PER_IMAGE, WORDS_NUM=args.WORDS_NUM, transform=train_tf)
assert dataset_train
train_set = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size,
    drop_last=True, shuffle=True, num_workers=args.n_threads)

print(len(train_set))

ixtoword_train = dataset_train.ixtoword

g_model = InpaintNet().to(device)
pd_model = PatchDiscriminator().to(device)

params = list(g_model.parameters())  # g_model is model name
k = 0
for i in params:
    l = 1
    print('The construction of this layer param:' + str(list(i.size())))
    for j in i.size():
        l *= j
    print('The number of this layer param:' + str(l))
    k = k + l
print('The total count of this layer param ' + str(k))

print('________________________________________')

params = list(pd_model.parameters())  # g_model is model name
k = 0
for i in params:
    l = 1
    print('The construction of this layer param:' + str(list(i.size())))
    for j in i.size():
        l *= j
    print('The number of this layer param:' + str(l))
    k = k + l
print('The total count of this layer param ' + str(k))

l1 = nn.L1Loss().to(device)
change_size = nn.Conv2d(32, 3, 1, 1, padding=0).to(device)

g_optimizer_t = torch.optim.Adam(
    g_model.parameters(),
    args.lr, (args.b1, args.b2))
pd_optimizer_t = torch.optim.Adam(
    pd_model.parameters(),
    args.lr, (args.b1, args.b2))

if args.resume:
    g_checkpoint = torch.load(save_dir + '/G_' + str(args.resume) + '.pth', map_location=device)
    g_model.load_state_dict(g_checkpoint)
    pd_checkpoint = torch.load(save_dir + '/PD_' + str(args.resume) + '.pth', map_location=device)
    pd_model.load_state_dict(pd_checkpoint)
    print('Models restored')

image_encoder = CNN_ENCODER(args.image_size)
img_encoder_path = '../../DAMSMencoders/' + args.dataset + '/image_encoder500.pth'
state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
image_encoder.load_state_dict(state_dict)
for p in image_encoder.parameters():
    p.requires_grad = False
print('Load image encoder from:', img_encoder_path)
image_encoder.eval()

print('dataset_train.n_words', dataset_train.n_words)

text_encoder = RNN_ENCODER(dataset_train.n_words, nhidden=args.image_size)
text_encoder_path = '../../DAMSMencoders/' + args.dataset + '/text_encoder500.pth'
state_dict = torch.load(text_encoder_path, map_location=lambda storage, loc: storage)
text_encoder.load_state_dict(state_dict)
for p in text_encoder.parameters():
    p.requires_grad = False
print('Load text encoder from:', text_encoder_path)
text_encoder.eval()

if use_cuda:
    text_encoder = text_encoder.cuda()
    image_encoder = image_encoder.cuda()


def prepare_labels():
    batch_size = args.batch_size
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
    match_labels = Variable(torch.LongTensor(range(batch_size)))
    if use_cuda:
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()
        match_labels = match_labels.cuda()

    return real_labels, fake_labels, match_labels


real_labels, fake_labels, match_labels = prepare_labels()


def get_mask():
    mask = []
    IMAGE_SIZE = args.image_size

    for i in range(args.batch_size):
        q1 = p1 = IMAGE_SIZE // 4
        q2 = p2 = IMAGE_SIZE - IMAGE_SIZE // 4

        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        m[q1:q2 + 1, p1:p2 + 1] = 1
        m = np.expand_dims(m, 0)
        mask.append(m)

    mask = np.array(mask)
    mask = torch.from_numpy(mask)

    if use_cuda:
        mask = mask.float().cuda()
    return mask


def local_patch(location, pix_num):
    local = []
    IMAGE_SIZE = args.image_size
    l = location[0] - pix_num
    r = location[2] + 1 + pix_num
    t = location[3] - pix_num
    d = location[1] + 1 + pix_num
    if l < 0:
        l = 0
    if r > 255:
        r = 255
    if t < 0:
        t = 0
    if d > 255:
        d = 255
    m = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    m[l:r, t:d] = 1
    m = np.expand_dims(m, 0)
    local.append(m)

    local = np.array(local)
    local = torch.from_numpy(local)

    if use_cuda:
        local = local.float().cuda()
    return local


def prepare_local(local_file, file_name):
    norm = transforms.Compose([transforms.ToTensor()])
    img = Image.open(local_file + '/all_mask/' + file_name).convert('RGB')
    if torch.cuda.is_available():
        mask = Variable(norm(img)).cuda()
    else:
        mask = (Variable(norm(img)))
    return mask


nz = 100
noise = Variable(torch.FloatTensor(args.batch_size, nz))
fixed_noise = Variable(torch.FloatTensor(args.batch_size, nz).normal_(0, 1))
if use_cuda:
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

for i in range(args.start_epoch, args.max_epoch + 1):

    iterator_train = iter(train_set)
    train_step = 1
    num_batches = len(train_set) + 1
    while train_step < num_batches:

        data_train = iterator_train.next()
        real_masks, imgs, captions, cap_lens, class_ids, keys = prepare_data(data_train)

        hidden = text_encoder.init_hidden(args.batch_size)
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        text_mask = (captions == 0)
        num_words = words_embs.size(2)
        if text_mask.size(1) > num_words:
            text_mask = text_mask[:, :num_words]

        img = imgs[-1]
        real_mask = real_masks[-1]

        if args.mask_end_num == 0:
            mask = get_mask()
            mask_local = local_patch([63, 191, 191, 63], args.LD)
            image_local = img * mask_local
        else:
            mask_int = random.randint(args.mask_start_num + 1, args.mask_end_num + 1)
            mask_str = str(mask_int // 10000) + str((mask_int % 10000) // 1000) + str((mask_int % 1000) // 100) + str(
                (mask_int % 100) // 10) + str((mask_int % 10))
            file_name = 'mask_' + mask_str + '.jpg'
            with open(os.path.join(args.mask_file, 'boundary.json'), 'rb') as f:
                local_list = json.load(f)
            mask_before = prepare_local(args.mask_file, file_name)
            mask = (1. - mask_before)
            mask_local = local_patch(local_list.get(file_name), args.LD)
            image_local = img * mask_local

        masked = img * (1. - mask)
        noise.data.normal_(0, 1)
        
        coarse_result_t, refine_result_t1, refine_result_t2, attn_loss, attn_refine_loss1, attn_refine_loss2, h_code, h_code_out1, h_code_out2 = g_model(
            masked, mask, noise, sent_emb, words_embs, text_mask)

        ################### Text-Image Matching Loss ####################
        region_features, cnn_code = image_encoder(refine_result_t2)
        w_loss0, w_loss1, _ = words_loss(region_features, words_embs, match_labels, cap_lens, class_ids,
                                         args.batch_size, use_cuda)
        w_loss = (w_loss0 + w_loss1) * 1.0

        s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb, match_labels, class_ids, args.batch_size, use_cuda)
        s_loss = (s_loss0 + s_loss1) * 1.0

        matching_loss = w_loss + s_loss

        ################### D loss ####################
        pd_loss_t = calc_D_loss(pd_model, refine_result_t2, img, mask_local, args.LD)
        pd_optimizer_t.zero_grad()
        pd_loss_t.backward(retain_graph=True)
        pd_optimizer_t.step()

        ################### G loss ####################
        pg_loss_t = calc_G_loss(pd_model, refine_result_t2, img, mask_local, args.LD)

        recon_loss = l1(coarse_result_t, img) + l1(refine_result_t1, img) + l1(refine_result_t2, img)

        recon_loss_attn = l1(coarse_result_t * attn_loss, img * attn_loss) + l1(refine_result_t1 * attn_refine_loss1, img * attn_refine_loss1) + l1(refine_result_t2 * attn_refine_loss2, img * attn_refine_loss2)

        recon_loss_mask = l1(coarse_result_t * real_mask, img * real_mask) + l1(refine_result_t1 * real_mask, img * real_mask) + l1(refine_result_t2 * real_mask, img * real_mask)

        total_loss_t = args.Rec * recon_loss + args.TGA * recon_loss_attn + args.M * recon_loss_mask + args.TIM * matching_loss + args.G * pg_loss_t

        g_optimizer_t.zero_grad()
        total_loss_t.backward()
        g_optimizer_t.step()

        num_save_interval = num_batches * i + train_step

        if num_save_interval % args.save_model_interval == 0 or i == args.max_epoch:
            torch.save(g_model.state_dict(), f'{save_dir}/G_{num_save_interval}.pth')
            torch.save(pd_model.state_dict(), f'{save_dir}/PD_{num_save_interval}.pth')
            print("model saved.")
            if i == args.max_epoch:
                break

        if num_save_interval % args.log_interval == 0:
            print('\n', num_save_interval)
            print('g_loss_t/total_loss_t', total_loss_t.item())
            print('g_loss_t/recon_loss', args.Rec * recon_loss.item())
            print('g_loss_t/recon_loss_attn', args.TGA * recon_loss_attn.item())
            print('g_loss_t/recon_loss_mask', args.M * recon_loss_mask.item())
            print('g_loss_t/matching_loss', args.TIM * matching_loss.item())
            print('g_loss_t/pg_loss_t', args.G * pg_loss_t.item())
            print('d_loss_t/pd_loss_t', pd_loss_t.item())


        def denorm(x):
            out = (x + 1) / 2  # [-1,1] -> [0,1]
            return out.clamp_(0, 1)


        if num_save_interval % args.vis_interval == 0:
            ims = torch.cat(
                [masked, coarse_result_t, refine_result_t1, refine_result_t2, img, image_local, refine_result_t2 * mask,
                 img * real_mask, refine_result_t2 * real_mask, change_size(h_code),
                 change_size(h_code_out1), change_size(h_code_out2)], dim=3)
            ims_train = ims.add(1).div(2).mul(255).clamp(0, 255).byte()
            ims_train = ims_train[0].permute(1, 2, 0).data.cpu().numpy()

            cap_back = Image.new('RGB', (ims_train.shape[1], 30), (255, 255, 255))
            cap = captions[0].data.cpu().numpy()
            sentence = []
            for j in range(len(cap)):
                if cap[j] == 0:
                    break
                word = ixtoword_train[cap[j]].encode('ascii', 'ignore').decode('ascii')
                sentence.append(word)
            sentence = ' '.join(sentence)
            draw = ImageDraw.Draw(cap_back)
            draw.text((0, 10), sentence, (0, 0, 0))
            cap_back = np.array(cap_back)

            ims_text = np.concatenate([ims_train, cap_back], 0)
            ims_out = Image.fromarray(ims_text)
            fullpath = '%s/epoch%d_iteration%d.png' % (training_image, i, num_save_interval)
            ims_out.save(fullpath)

            print("train image saved.")

        train_step = train_step + 1
