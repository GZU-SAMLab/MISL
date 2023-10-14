import numpy as np
import torch
from torch import nn

from GlobalAttention import func_attention


def calc_G_loss(discriminator, output, target, real_mask, LD):
    y_pred_fake, y_pred_fake_local = discriminator(output, target, real_mask)
    y_pred, y_pred_local = discriminator(target, output, real_mask)

    g_loss = (torch.mean((y_pred - torch.mean(y_pred_fake) + 1.) ** 2) + torch.mean(
        (y_pred_fake - torch.mean(y_pred) - 1.) ** 2)) / 2
    if LD != -1:
        g_loss_local = (torch.mean((y_pred_local - torch.mean(y_pred_fake_local) + 1.) ** 2) + torch.mean(
            (y_pred_fake_local - torch.mean(y_pred_local) - 1.) ** 2)) / 2
        return g_loss + g_loss_local
    else:
        return g_loss

def calc_D_loss(discriminator, output, target, real_mask, LD):
    y_pred_fake, y_pred_fake_local = discriminator(output, target, real_mask)
    y_pred, y_pred_local = discriminator(target, output, real_mask)

    d_loss = (torch.mean((y_pred - torch.mean(y_pred_fake) - 1.) ** 2) + torch.mean(
        (y_pred_fake - torch.mean(y_pred) + 1.) ** 2)) / 2
    if LD != -1:
        d_loss_local = (torch.mean((y_pred_local - torch.mean(y_pred_fake_local) - 1.) ** 2) + torch.mean(
            (y_pred_fake_local - torch.mean(y_pred_local) + 1.) ** 2)) / 2
        return d_loss + d_loss_local
    else:
        return d_loss


# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels, class_ids, batch_size, use_cuda=True, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if use_cuda:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * 10.0

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()  # [1, 6, 6] -> [6, 6]
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels, cap_lens, class_ids, batch_size, use_cuda=True, ):
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features

        weiContext, attn = func_attention(word, context, 5.0)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        row_sim.mul_(5.0).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if use_cuda:
            masks = masks.cuda()

    similarities = similarities * 10.0
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


def cmd_loss(x1, x2, n_moments):
    def matchnorm(x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return matchnorm(ss1, ss2)

    mx1 = torch.mean(x1, 0)
    mx2 = torch.mean(x2, 0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = matchnorm(mx1, mx2)
    scms = dm
    for i in range(n_moments - 1):
        scms += scm(sx1, sx2, i + 2)
    return scms
