import os

import numpy
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from torchvision.models.inception import inception_v3
from skimage.transform import resize
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from scipy.stats import entropy


def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    def collate_function(data):
        """
        :data: a list for a batch of samples. [[string, tensor], ..., [string, tensor]]
        """

        id, imgs = list(zip(*data))
        # directorys, imgs = transposed_data[0], transposed_data[1]
        imgs = torch.stack(imgs, 0)
        return id, imgs

    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, )

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))
    id = 0
    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        id = id + 1
        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)
        if i % 10 == 0:
            print(i)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            # plt.imshow(self.orig[index][0])
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.transforms as transforms
    import torchvision.transforms as transforms

    for i in range(24, 40):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('./image/real' + str(i * 10000))
        print('yuan' + str(i * 10000))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        train_dir = './images/birds/only' + str(i * 10000)  # 注意此处不能使用subdir，因为之后的某些值在test及val中也需要使用
        all_pics = os.listdir(train_dir)
        images_list = list()
        for image in all_pics:
            img = Image.open(train_dir + '/' + image).convert('RGB')
            trans = transforms.Compose([transforms.ToTensor()])
            img = trans(img)
            # resize with nearest neighbor interpolation
            new_image = resize(img, (3, 256, 256), 0)
            # store
            images_list.append(new_image)
        mean, std = inception_score(np.asarray(images_list), cuda=True, batch_size=32, resize=True, splits=10)
        print('np.mean(split_scores) : %f , np.std(split_scores) : %f' % (mean, std))
        txt_file = open('./MMFL/IS.txt', 'a+')
        txt_file.write('yuan' + str(i * 10000))
        txt_file.write('\n')
        txt_file.write(str(time.asctime(time.localtime(time.time()))))
        txt_file.write('\n')
        txt_file.write('np.mean(split_scores) : %f , np.std(split_scores) : %f' % (mean, std))
        txt_file.write('\n' * 2)


