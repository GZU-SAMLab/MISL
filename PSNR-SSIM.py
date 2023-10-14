import os
import cv2
import time
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def calc_measures(hr_path, fake_path, name, file_name, calc_psnr=True, calc_ssim=True):
    HR_files = os.listdir(hr_path)
    mean_psnr = 0
    mean_ssim = 0
    num = 0

    for file in HR_files:
        path_hr = os.path.join(hr_path, file)
        hr_img = cv2.imread(path_hr, 1)
        path = os.path.join(fake_path, file)
        if not os.path.isfile(path):
            raise FileNotFoundError('')

        inf_img = cv2.imread(path, 1)

        num += 1
        if num % 100 == 0:
            print(num)

        if calc_psnr:
            psnr = compare_psnr(hr_img, inf_img)
            mean_psnr += psnr
        if calc_ssim:
            ssim = compare_ssim(hr_img, inf_img, multichannel=True)
            mean_ssim += ssim

    print('-' * 10)
    if calc_psnr:
        M_psnr = mean_psnr / len(HR_files)
        print('mean-PSNR %f dB' % M_psnr)
    if calc_ssim:
        M_ssim = mean_ssim / len(HR_files)
        print('mean-SSIM %f' % M_ssim)

    txt_file = open(file_name, 'a+')
    txt_file.write(name)
    txt_file.write('\n')
    txt_file.write(str(time.asctime(time.localtime(time.time()))))
    txt_file.write('\n')
    txt_file.write('mean-PSNR: %f , mean-SSIM: %f' % (M_psnr, M_ssim))
    txt_file.write('\n' * 2)


if __name__ == '__main__':
    for i in range(20, 21):  # birds flowers celeba15000  coco
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('birds_mask-0-15000_M-0_LD--1-' + str(i * 10000))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        calc_measures('./images/birds_mask-0-15000_M-0_LD--1/real',
                      './images/birds_mask-0-15000_M-0_LD--1/only/' + str(i * 10000),
                      str(i * 10000), 'result/birds_mask-0-15000_M-0_LD--1-psnr_ssim.txt',
                      calc_psnr=True, calc_ssim=True)
