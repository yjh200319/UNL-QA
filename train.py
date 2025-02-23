import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import time
import os

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

from dataset import Train_DwiData, Val_DwiData
from model import VectorQuantizedVAE

# Hyperparameters
kl_weight = 1
num_epochs = 2
train_batch_size = 32
val_batch_size = 32
learning_rate = 1e-4
beta = 1.0
num_channels = 100
hidden_size = 256
k = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def loss_fn(y, y_hat, mean, logvar):
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)
    loss = recons_loss + kl_loss * kl_weight
    return loss


# 统计一个batch的总共的结构相似度SSIM
def calculate_ssim(origin_img, recon_img):
    f_batch_size = origin_img.size(0)
    total_ssim = 0

    for i in range(f_batch_size):
        # 将张量转换为numpy数组，并调整维度
        img_np = origin_img[i].squeeze().cpu().detach().numpy()
        x_tilde_np = recon_img[i].squeeze().cpu().detach().numpy()

        # 计算SSIM
        ssim_value = ssim(img_np, x_tilde_np, data_range=img_np.max() - img_np.min())

        total_ssim += ssim_value

    # mean_ssim = total_ssim / batch_size
    return total_ssim


if not os.path.exists(f'./Train_epoch={num_epochs}'):
    os.mkdir(f'./Train_epoch={num_epochs}')

if not os.path.exists(f'./Val_epoch={num_epochs}'):
    os.mkdir(f'./Val_epoch={num_epochs}')

if not os.path.exists(f'./checkpoint_MSE'):
    os.mkdir(f'./checkpoint_MSE')

if not os.path.exists(f'./checkpoint_SSIM'):
    os.mkdir(f'./checkpoint_SSIM')

if not os.path.exists(f'./Train-image'):
    os.mkdir(f'./Train-image')

if not os.path.exists(f'./Val-image'):
    os.mkdir(f'./Val-image')

img_transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# train_folder_path = '/media/UG1/wtl/Caffine'
train_folder_path = '/data/yjh/Split_Caffine/Train'
val_folder_path = '/data/yjh/Split_Caffine/Validate'
# #######加载训练集和验证集###############
start_time = time.time()

# 取z=24、25、26中脑切片
train_data_set1 = Train_DwiData(train_folder_path, 24)
train_data_set2 = Train_DwiData(train_folder_path, 25)
train_data_set3 = Train_DwiData(train_folder_path, 26)

train_data_set = train_data_set1 + train_data_set2 + train_data_set3
print("***************" * 3 + "训练集所有z=24,25,26的中脑数据加载完成" + "***************" * 3)
val_data_set = Val_DwiData(val_folder_path)
# ###########加载训练数据 + 验证数据
train_dataloader = DataLoader(train_data_set, batch_size=train_batch_size, shuffle=True)
val_dataloader = DataLoader(val_data_set, batch_size=val_batch_size, shuffle=True)
# ############################## ####

model = VectorQuantizedVAE(num_channels, hidden_size, k).to(device)
if torch.cuda.is_available():
    # model.cuda()
    print('cuda is OK!')
    model = model.to('cuda')
else:
    print('cuda is NO!')

reconstruction_function = nn.MSELoss(size_average=False)

# reconstruction_function = nn.MSELoss(reduction=sum)
# def loss_function(recon_x, x, mu, logvar):
#     """
#     recon_x: generating images
#     x: origin images
#     mu: latent mean
#     logvar: latent log variance
#     """
#     BCE = reconstruction_function(recon_x, x)  # mse loss
#     # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
#     KLD = torch.sum(KLD_element).mul_(-0.5)
#     # KL divergence
#     return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-4)
avg_train_loss = []
avg_val_loss = []
best_loss = -1
best_epoch = -1
best_ssim = -1
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_epoch_ssim = 0
    for batch_idx, data in enumerate(train_dataloader):
        img, _ = data
        # img = img.view(img.size(0), -1)
        # img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(img)
        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, img)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + beta * loss_commit
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        train_batch_total_ssim = calculate_ssim(img, x_tilde)
        train_epoch_ssim += train_batch_total_ssim
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f} \t SSIM: {:.5f}'.format(
                epoch,
                batch_idx * len(img),
                len(train_dataloader.dataset), 100. * batch_idx / len(train_dataloader),
                # loss.data[0] / len(img)))
                loss.item() / len(img), train_batch_total_ssim / len(img)))

    if epoch % 20 == 0:
        # realimage, _ = next(iter(dataloader))
        # new_realimage = realimage.view(realimage.size(0), -1)  # 变成[64.112*112]
        # fakeimage, _, _ = model(new_realimage)
        # [64,1,96,96]
        # 真假图像合并成一张
        image = torch.cat([img, x_tilde], dim=0)
        # image=image.cpu().data
        save_image(image, './Train_epoch={}/image_{}.png'.format(num_epochs, epoch))

        # 将图片的差值打出来看看
        diff = img - x_tilde  # 计算差值
        abs_diff = torch.abs(diff)  # 取绝对值
        sum_abs_diff = torch.sum(abs_diff, dim=(1, 2, 3))
        sample1 = sum_abs_diff[0].cpu().detach().numpy()

        # plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img[0].squeeze().cpu().detach().numpy(), cmap='gray')
        plt.title("Train-Original Fig")  # 使用 title 方法设置子标题
        plt.subplot(1, 2, 2)
        plt.imshow(x_tilde[0].squeeze().detach().cpu().numpy(), cmap='gray')
        plt.title("Train-Reconstruction Fig")  # 使用 title 方法设置子标题
        plt.suptitle(f"Signal subtraction: {sample1}")
        plt.savefig(f"./Train-image/Train-epoch={epoch}.png")
        plt.show()  # 将 show 方法放在 savefig 方法前面
    # ########################验证集##########################
    model.eval()
    val_loss = 0
    val_epoch_ssim = 0
    with torch.no_grad():
        for batch_idx1, data1 in enumerate(val_dataloader):
            img1, _ = data1
            # img = img.view(img.size(0), -1)
            # img = Variable(img)
            if torch.cuda.is_available():
                img1 = img1.cuda()
            optimizer.zero_grad()
            x_tilde1, z_e_x1, z_q_x1 = model(img1)
            # Reconstruction loss
            loss_recons1 = F.mse_loss(x_tilde1, img1)
            # Vector quantization objective
            loss_vq1 = F.mse_loss(z_q_x1, z_e_x1.detach())
            # Commitment objective
            loss_commit1 = F.mse_loss(z_e_x1, z_q_x1.detach())

            loss1 = loss_recons1 + loss_vq1 + beta * loss_commit1
            # loss.backward()
            # optimizer.step()
            val_loss += loss1.item()

            val_batch_total_ssim = calculate_ssim(img1, x_tilde1)
            val_epoch_ssim += val_batch_total_ssim

            if batch_idx1 % 50 == 0:
                print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}\t SSIM: {:.5f}'.format(
                    epoch,
                    batch_idx1 * len(img1),
                    len(val_dataloader.dataset), 100. * batch_idx1 / len(val_dataloader),
                    # loss.data[0] / len(img)))
                    loss1.item() / len(img1), val_batch_total_ssim / len(img1)))

    if epoch % 20 == 0:
        # realimage, _ = next(iter(dataloader))
        # new_realimage = realimage.view(realimage.size(0), -1)  # 变成[64.112*112]
        # fakeimage, _, _ = model(new_realimage)
        # [64,1,96,96]
        # 真假图像合并成一张
        image1 = torch.cat([img1, x_tilde1], dim=0)
        # image=image.cpu().data
        save_image(image1, './Val_epoch={}/image_{}.png'.format(num_epochs, epoch))

        # ##打印差值
        diff = img1 - x_tilde1  # 计算差值
        abs_diff = torch.abs(diff)  # 取绝对值
        sum_abs_diff = torch.sum(abs_diff, dim=(1, 2, 3))
        sample1 = sum_abs_diff[0].cpu().detach().numpy()

        # plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img1[0].squeeze().detach().cpu().numpy(), cmap='gray')
        plt.title("Val-Original Fig")  # 使用 title 方法设置子标题
        plt.subplot(1, 2, 2)
        plt.imshow(x_tilde1[0].squeeze().detach().cpu().numpy(), cmap='gray')
        plt.title("Val-Reconstruction Fig")  # 使用 title 方法设置子标题
        plt.suptitle(f"Signal subtraction:{sample1}")
        plt.savefig(f"./Val-image/Val-epoch={epoch}.png")
        plt.show()  # 将 show 方法放在 savefig 方法前面

    print(
        '====> Epoch: {} Average  Train loss: {:.10f} Train SSIM: {:.5f}  Validate loss: {:.10f}  Validate SSIM: {:.5f} '.format(
            epoch, train_loss / len(train_dataloader.dataset), train_epoch_ssim / len(train_dataloader.dataset),
                   val_loss / len(val_dataloader.dataset), val_epoch_ssim / len(val_dataloader.dataset)))
    avg_train_loss.append(train_loss / len(train_dataloader.dataset))
    avg_val_loss.append(val_loss / len(val_dataloader.dataset))

    if (epoch == 0) or (val_loss / len(val_dataloader.dataset) < best_loss):
        best_loss = val_loss / len(val_dataloader.dataset)
        best_epoch = epoch
        torch.save(model.state_dict(), f'./checkpoint_MSE/best_model.pth')

    if (epoch == 0) or (val_epoch_ssim / len(val_dataloader.dataset) > best_ssim):
        best_ssim = val_epoch_ssim / len(val_dataloader.dataset)
        best_epoch = epoch
        torch.save(model.state_dict(), f'./checkpoint_SSIM/best_model.pth')

    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), f'./checkpoint_MSE/model_{epoch + 1}.pth')
        torch.save(model.state_dict(), f'./checkpoint_SSIM/model_{epoch + 1}.pth')
# torch.save(model.state_dict(), f'./vae_epoch={num_epochs}.pth')
end_time = time.time()
# 计算训练时长（以分钟为单位）
duration_minutes = (end_time - start_time) / 60
print("模型训练完成！ best_model 出现在: epoch=" + str(best_epoch + 1) + " best_loss=" + str(best_loss))
print("训练时长（分钟）：", duration_minutes, "min")

plt.figure()
plt.plot(avg_train_loss, label='Train_loss')
plt.plot(avg_val_loss, label='Val_loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average_loss')
plt.title("Loss(Recons+VQ+Commit) VS Epoch")
plt.savefig('train_val_loss.png')
plt.show()
