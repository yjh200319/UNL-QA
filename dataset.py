import torch
import os.path
from dipy.io.image import load_nifti
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from os.path import join
from dipy.io import read_bvals_bvecs
from torchvision import transforms
import matplotlib.animation as animation


def normalization(data):
    mean = np.mean(data)
    std = np.std(data)

    return (data - mean) / std


class Train_DwiData(Dataset):

    # /media/overwater/E/NIMG_data/Caffine/sub-015  返回归一化后的dwi（b=1000）
    def select(self, sub_path, slice_num):
        ###############################
        dwi, affine = load_nifti(join(sub_path, 'dwi.nii.gz'), return_img=False)
        print("已经加载完成dwi.nii.gz")
        print(dwi.shape)
        # dwi * mask
        mask, _ = load_nifti(join(sub_path, 'dwi_mask.nii.gz'), return_img=False)
        mask = np.expand_dims(mask, axis=-1)
        dwi *= mask

        print("已经做完dwi*mask操作")
        print("Having loaded the subject: " + sub_path[-7:])

        bvals, bvecs = read_bvals_bvecs(join(sub_path, 'dwi.bval'), join(sub_path, 'dwi.bvec'))

        bvals = np.round(bvals / 100) * 100

        print(bvals)
        # b = 0  return type is tuple with a length of 1.So take the first element
        # 获取b=0的索引
        indices1 = np.where(bvals == 0)[0]
        print(indices1)

        # b = 1000
        indices2 = np.where(bvals == 1000)[0]
        print(indices2)

        # b = 3000
        indices3 = np.where(bvals == 3000)[0]
        print(indices3)

        # get sphere mean signal 0 1000 3000
        # b=0的乘过mask后的球面平均值
        # 对[112,112,50,8]中的最后一个维度8求平均信号值->得到[112,112,50]->这里要扩充维度方便后面广播
        s0 = np.mean(dwi[..., indices1], axis=3)

        print(s0.shape)
        # 从[112,112,50]->[112,112,50]->[112,112,50,1]
        s0 = np.expand_dims(s0, axis=-1)

        # # divide
        dwi[..., indices2] = - np.log(dwi[..., indices2] / s0) / 1000
        dwi[..., indices3] = - np.log(dwi[..., indices3] / s0) / 3000

        # nan全部设为0
        dwi = np.where(np.isnan(dwi), 0, dwi)

        # save res
        print(dwi.shape)
        # 每个volume单独归一化
        for v in range(dwi.shape[3]):
            dwi[..., v] = normalization(dwi[..., v])
        ############################################
        # dwi, _ = load_nifti(join(sub_path, 'dwi.nii.gz'), return_img=False)
        # bval, bvec = read_bvals_bvecs(join(sub_path, 'dwi.bval'), join(sub_path, 'dwi.bvec'))

        # trim
        # 选取
        indices = np.concatenate((indices2, indices3))
        # indices = indices2 + indices3
        dwi = dwi[:, :, slice_num, indices]
        dwi = np.expand_dims(dwi, axis=2)

        # dwi = dwi[:, :, :, indices]
        # dwi = np.expand_dims(dwi, axis=2)

        #
        print(np.max(dwi))
        # normalization
        dwi = dwi / np.max(dwi)
        # print(np.mean(dwi))
        print("针对样本：" + sub_path[-7:] + "消除完b值影响")
        return dwi

    # data folder  such as ......./BTC ...../Caffine  ..../HCP
    def __init__(self, folder_path, slice_num):
        self.folder_path = folder_path
        self.slice_num = slice_num
        all_dwis = []

        # count
        count = 0
        # traverse 遍历指定文件夹下所有subject
        for subject in sorted(os.listdir(folder_path), reverse=False):
            print(join(folder_path, subject))
            # 得到 一个subject的dwi数据
            sub_dwi = self.select(join(folder_path, subject), slice_num)
            print(sub_dwi.shape)
            all_dwis.append(sub_dwi)
            count += 1
            if count == 100:
                break

        # count
        print(f"一共有{len(all_dwis)}个subject")

        # 在第3维 合并 112 112 50 32*n
        all_data = np.concatenate(all_dwis, axis=3)
        print(all_data.shape)

        # 将data变成tensor
        torch_data = torch.FloatTensor(all_data)
        self.flatten_3d = torch_data.permute(3, 2, 1, 0).contiguous().view(all_data.shape[2] * all_data.shape[3], 1,
                                                                           112, 112)
        print(f"最终dataset数据的格式为：{self.flatten_3d.shape}")

    def __getitem__(self, idx):
        img = self.flatten_3d[idx]
        return img, idx

    def __len__(self):
        # 一个例如BTC或者Caffine文件夹里一共有多少个患者数据
        return len(self.flatten_3d)


class Val_DwiData(Dataset):

    # /media/overwater/E/NIMG_data/Caffine/sub-015  返回归一化后的dwi（b=1000）
    def select(self, sub_path):
        ###############################
        dwi, affine = load_nifti(join(sub_path, 'dwi.nii.gz'), return_img=False)
        print("已经加载完成dwi.nii.gz")
        print(dwi.shape)
        # dwi * mask
        mask, _ = load_nifti(join(sub_path, 'dwi_mask.nii.gz'), return_img=False)
        mask = np.expand_dims(mask, axis=-1)
        dwi *= mask

        print("已经做完dwi*mask操作")
        print("Having loaded the subject: " + sub_path[-7:])

        bvals, bvecs = read_bvals_bvecs(join(sub_path, 'dwi.bval'), join(sub_path, 'dwi.bvec'))

        bvals = np.round(bvals / 100) * 100

        print(bvals)
        # b = 0  return type is tuple with a length of 1.So take the first element
        # 获取b=0的索引
        indices1 = np.where(bvals == 0)[0]
        print(indices1)

        # b = 1000
        indices2 = np.where(bvals == 1000)[0]
        print(indices2)

        # b = 3000
        indices3 = np.where(bvals == 3000)[0]
        print(indices3)

        # get sphere mean signal 0 1000 3000
        # b=0的乘过mask后的球面平均值
        # 对[112,112,50,8]中的最后一个维度8求平均信号值->得到[112,112,50]->这里要扩充维度方便后面广播
        s0 = np.mean(dwi[..., indices1], axis=3)

        print(s0.shape)
        # 从[112,112,50]->[112,112,50]->[112,112,50,1]
        s0 = np.expand_dims(s0, axis=-1)

        # # divide
        dwi[..., indices2] = - np.log(dwi[..., indices2] / s0) / 1000
        dwi[..., indices3] = - np.log(dwi[..., indices3] / s0) / 3000

        # nan全部设为0
        dwi = np.where(np.isnan(dwi), 0, dwi)

        # save res
        print(dwi.shape)
        # 每个volume单独归一化
        for v in range(dwi.shape[3]):
            dwi[..., v] = normalization(dwi[..., v])
        ############################################
        # dwi, _ = load_nifti(join(sub_path, 'dwi.nii.gz'), return_img=False)
        # bval, bvec = read_bvals_bvecs(join(sub_path, 'dwi.bval'), join(sub_path, 'dwi.bvec'))

        # trim
        # 选取
        indices = np.concatenate((indices2, indices3))
        # indices = indices2 + indices3
        dwi = dwi[:, :, 25, indices]
        dwi = np.expand_dims(dwi, axis=2)

        # dwi = dwi[:, :, :, indices]
        # dwi = np.expand_dims(dwi, axis=2)

        #
        print(np.max(dwi))
        # normalization
        dwi = dwi / np.max(dwi)
        # print(np.mean(dwi))
        print("针对样本：" + sub_path[-7:] + "消除完b值影响")
        return dwi

    # data folder  such as ......./BTC ...../Caffine  ..../HCP
    def __init__(self, folder_path):
        self.folder_path = folder_path
        all_dwis = []

        # count
        count = 0
        # traverse 遍历指定文件夹下所有subject
        for subject in sorted(os.listdir(folder_path), reverse=False):
            print(join(folder_path, subject))
            # 得到 一个subject的dwi数据
            sub_dwi = self.select(join(folder_path, subject))
            print(sub_dwi.shape)
            all_dwis.append(sub_dwi)
            count += 1
            if count == 20:
                break

        # count
        print(f"一共有{len(all_dwis)}个subject")

        # 在第3维 合并 112 112 50 32*n
        all_data = np.concatenate(all_dwis, axis=3)
        print(all_data.shape)

        # 将data变成tensor
        torch_data = torch.FloatTensor(all_data)
        self.flatten_3d = torch_data.permute(3, 2, 1, 0).contiguous().view(all_data.shape[2] * all_data.shape[3], 1,
                                                                           112, 112)
        print(f"最终dataset数据的格式为：{self.flatten_3d.shape}")

    def __getitem__(self, idx):
        img = self.flatten_3d[idx]
        return img, idx

    def __len__(self):
        # 一个例如BTC或者Caffine文件夹里一共有多少个患者数据
        return len(self.flatten_3d)


if __name__ == '__main__':
    folder_path = '/media/overwater/E/NIMG_data/Caffine'
    dwi_data = DwiData(folder_path)

    num_fig = 20
    for i in range(num_fig):
        img = plt.imshow(np.squeeze(dwi_data.flatten_3d[i]), cmap='gray')
        plt.show()
        plt.pause(0.5)
