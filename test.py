import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio

from utils3 import is_normality_with_KS_test, \
    visualize_sigmoid_probabilities
from dataset import Val_dHCP_DwiData, Test_dHCP_DwiData
from utils1 import mse_mean_variance_dual_line_classification, ssim_mean_variance_dual_line_classification, \
    psnr_mean_variance_dual_line_classification, check_normality_with_KS_test, check_normality_with_SW_test
from model import VectorQuantizedVAE
import random
from scipy.special import expit  # Sigmoid 函数
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 设置随机数种子，保证实验可复现
num_channels = 1
hidden_size = 256
seed = 66
random.seed(seed)
torch.manual_seed(seed)


# ###########验证集函数#####################
# #####根据dataloader进行前向传播#######
def Validate(dataloader):
    print("***" * 3 + "验证开始！" + "***" * 3)
    mse_list = []
    ssim_list = []
    psnr_list = []
    for batch_idx, data in enumerate(dataloader):
        img, _ = data  # [4,1,112,112]
        # print(img.shape)
        # img = img2.view(img2.size(0), -1)  # [4,112*112]
        # recon_batch, mu, logvar = model(img)  # [4,112*112]
        recon_batch_img, z_e_x, z_q_x = model(img)

        # ##### 计算重建前后的MSE误差#####
        mse_loss = MSE(img, recon_batch_img)
        mse_list.append(mse_loss)

        # ###### 计算重建前后的SSIM误差###########
        new_img = img.squeeze().cpu().numpy()
        new_rec = recon_batch_img.squeeze().cpu().numpy()
        ssim_loss = ssim(new_img, new_rec, data_range=max(new_img.max(), new_rec.max()))
        ssim_list.append(ssim_loss)

        # 计算PSNR值
        psnr_value = peak_signal_noise_ratio(new_img, new_rec, data_range=new_img.max() - new_img.min())
        psnr_list.append(psnr_value)

        if batch_idx % 2 == 0:
            print('Validate batch_idx: [{}/{} ({:.0f}%)]\t'.format(
                batch_idx * len(img),
                len(dataloader.dataset), 100. * batch_idx / len(dataloader)))

            # # #####################将原图和生成图片拼接对比##############################
            # image = torch.cat([img, recon_batch_img], dim=0)
            # save_image(image, f'./vae_img_test_good/image_batch_idx_{batch_idx}.png')
            # print("测试集good图片：" + str(batch_idx + 1) + "批次图片完成！")
    plt.subplot(1, 2, 1).imshow(new_img, cmap='gray')
    plt.title("Before Reconstruction")
    plt.subplot(1, 2, 2).imshow(new_rec, cmap='gray')
    plt.title("After Reconstruction")
    plt.suptitle("Comparison of Input Slices And Output Slices")
    plt.show()
    print("***" * 3 + "该数据集验证完成！" + "***" * 3)
    return mse_list, ssim_list, psnr_list


# #########################测试###################################
def Test(dataloader):
    print("***" * 3 + "测试开始！" + "***" * 3)
    mse_list = []
    ssim_list = []
    psnr_list = []
    for batch_idx, data in enumerate(dataloader):
        img, _ = data  # [4,1,112,112]
        # print(img.shape)
        # img = img2.view(img2.size(0), -1)  # [4,112*112]
        # recon_batch, mu, logvar = model(img)  # [4,112*112]
        recon_batch_img, z_e_x, z_q_x = model(img)

        # ##### 计算重建前后的MSE误差#####
        mse_loss = MSE(img, recon_batch_img)
        mse_list.append(mse_loss)

        # ###### 计算重建前后的SSIM误差###########
        new_img = img.squeeze().cpu().numpy()
        new_rec = recon_batch_img.squeeze().cpu().numpy()
        ssim_loss = ssim(new_img, new_rec, data_range=max(new_img.max(), new_rec.max()))
        ssim_list.append(ssim_loss)

        # 计算PSNR值
        psnr_value = peak_signal_noise_ratio(new_img, new_rec, data_range=max(new_img.max(), new_rec.max()))
        psnr_list.append(psnr_value)

        if batch_idx % 2 == 0:
            print('Test batch_idx: [{}/{} ({:.0f}%)]\t'.format(
                batch_idx * len(img),
                len(dataloader.dataset), 100. * batch_idx / len(dataloader)))

            # # #####################将原图和生成图片拼接对比##############################
            # image = torch.cat([img, recon_batch_img], dim=0)
            # save_image(image, f'./vae_img_test_good/image_batch_idx_{batch_idx}.png')
            # print("测试集good图片：" + str(batch_idx + 1) + "批次图片完成！")
    plt.subplot(1, 2, 1).imshow(new_img, cmap='gray')
    plt.title("Before Reconstruction")
    plt.subplot(1, 2, 2).imshow(new_rec, cmap='gray')
    plt.title("After Reconstruction")
    plt.suptitle("Comparison of Input Slices And Output Slices")
    plt.show()
    print("***" * 3 + "该数据集测试完成！" + "***" * 3)
    return mse_list, ssim_list, psnr_list


# ############################################
# 从左到右分别是good、ghost、spike...
def mse_scatter_plot(list1, list2, list3, list4, list5):
    plt.scatter(range(len(list1)), list1, label='good')  # 绘制 MSE 的散点图
    plt.scatter(range(len(list2)), list2, label='ghost')
    plt.scatter(range(len(list3)), list3, label='swap')
    plt.scatter(range(len(list4)), list4, label='noise')
    plt.scatter(range(len(list5)), list5, label='motion')
    plt.xlabel('Slice num')
    plt.ylabel('MSE Values')
    plt.title('Comparison of Good MSE and Artifact MSE')
    plt.legend(loc='upper right')
    plt.show()


def ssim_scatter_plot(list1, list2, list3, list4, list5):
    plt.scatter(range(len(list1)), list1, label='good')  # 绘制 MSE 的散点图
    plt.scatter(range(len(list2)), list2, label='ghost')
    plt.scatter(range(len(list3)), list3, label='swap')
    plt.scatter(range(len(list4)), list4, label='noise')
    plt.scatter(range(len(list5)), list5, label='motion')
    plt.xlabel('Slice num')
    plt.ylabel('SSIM Values')
    plt.title('Comparison of Good SSIM and Artifact MSE')
    plt.legend()
    plt.show()


def psnr_scatter_plot(list1, list2, list3, list4, list5):
    plt.scatter(range(len(list1)), list1, label='good')  # 绘制 MSE 的散点图
    plt.scatter(range(len(list2)), list2, label='ghost')
    plt.scatter(range(len(list3)), list3, label='swap')
    plt.scatter(range(len(list4)), list4, label='noise')
    plt.scatter(range(len(list5)), list5, label='motion')
    plt.xlabel('Slice num')
    plt.ylabel('PSNR Values')
    plt.title('Comparison of Good PSNR and Artifact PNSR')
    plt.legend()
    plt.show()


# ##########单独 artifact vs good MSE-SSIM 散点图
def good_vs_artifact_mse_scatter_plot(good, artifact, artifact_type):
    plt.scatter(range(len(good)), good, label='good')  # 绘制 MSE 的散点图
    plt.scatter(range(len(artifact)), artifact, label=f'{artifact_type}')
    plt.xlabel('Slices Num')
    plt.ylabel('MSE Values')
    plt.title(f'Comparison of Good MSE and Artifact {artifact_type} MSE')
    plt.legend()
    plt.show()


def good_vs_artifact_ssim_scatter_plot(good, artifact, artifact_type):
    plt.scatter(range(len(good)), good, label='good')  # 绘制 MSE 的散点图
    plt.scatter(range(len(artifact)), artifact, label=f'{artifact_type}')
    plt.xlabel('Slices Num')
    plt.ylabel('SSIM Values')
    plt.title(f'Comparison of Good SSIM and Artifact {artifact_type} SSIM')
    plt.legend()
    plt.show()


def good_vs_artifact_psnr_scatter_plot(good, artifact, artifact_type):
    plt.scatter(range(len(good)), good, label='good')  # 绘制 MSE 的散点图
    plt.scatter(range(len(artifact)), artifact, label=f'{artifact_type}')
    plt.xlabel('Slices Num')
    plt.ylabel('PSNR Values')
    plt.title(f'Comparison of Good PSNR and Artifact {artifact_type} PSNR')
    plt.legend()
    plt.show()


def plot_two_data_shuffle_divide_with_threshold(data1, data2, artifact_type):
    # 合并数据
    combined_data = data1 + data2
    labels = ['Good'] * len(data1) + ['Artifact'] * len(data2)

    mean_combined_data = np.mean(combined_data)
    std_combined_data = np.std(combined_data)
    threshold = mean_combined_data - lamda * std_combined_data
    # 生成随机顺序的索引
    idx = np.random.permutation(len(combined_data))

    # 根据随机索引重新排列数据和标签
    shuffled_data = [combined_data[i] for i in idx]
    shuffled_labels = [labels[i] for i in idx]

    # 创建颜色标签
    colors = ['navy' if label == 'Good' else 'orange' for label in shuffled_labels]
    # 绘制散点图
    plt.scatter(range(len(shuffled_data)), shuffled_data, c=colors)
    plt.axhline(y=threshold, color='red', linestyle='--', label=rf'$\mu-{lamda}\sigma$')
    plt.legend(loc='upper right')
    # plt.title(f'Comparison of Standard and {artifact_type} data')
    plt.xlabel('Slice Num')
    plt.ylabel('SSIM Value')
    plt.savefig(f"{artifact_type}.jpg")
    # 显示图表
    plt.show()


def evaluate_unsupervised_performance(ssim_good, ssim_bad):
    # 计算均值和标准差
    total = ssim_good + ssim_bad
    mean, std = np.mean(total), np.std(total)

    # 计算阈值
    threshold = mean - lamda * std

    plt.axhline(y=threshold, color='red', linestyle='--', label=rf'Threshold $\mu-{lamda}\sigma$')

    # 添加标签和图例
    plt.scatter(range(len(ssim_good)), ssim_good, label='Standard data')
    plt.scatter(range(len(ssim_bad)), ssim_bad, label='Artifact data')
    plt.title('SSIM Distribution with Threshold')
    plt.xlabel('Slice Num')
    plt.ylabel('SSIM Values')
    plt.legend(loc='upper right')
    plt.show()

    # 判定异常点
    predicted_labels = [0 if ssim > threshold else 1 for ssim in ssim_good + ssim_bad]
    # 计算正确率
    true_labels = [0] * len(ssim_good) + [1] * len(ssim_bad)
    accuracy1 = np.sum(np.array(predicted_labels) == np.array(true_labels)) / len(true_labels)

    recall, fpr, precision, f1, cm = calculate_confusion_matrix_scores(true_labels, predicted_labels)
    print(f"阈值: {threshold}")
    print(f"根据该阈值划分正确率: {accuracy1 * 100:.2f}%")

    return threshold, accuracy1, recall, fpr, precision, f1, cm


def calculate_distance_and_probabilities_2(T, ssim_A, ssim_B):
    # 计算每个点到阈值的距离
    total_ssim = ssim_A + ssim_B
    distances = total_ssim - T
    # distances = np.array(sorted(distances, reverse=True))

    true_labels = [0] * len(ssim_A) + [1] * len(ssim_B)
    sorted_indices = np.argsort(-distances)
    distances = distances[sorted_indices]
    true_labels = np.array(true_labels)[sorted_indices]
    # distances = np.argsort(distances)
    # 使用Sigmoid函数计算概率
    probabilities = expit(k_sigmoid * distances)  # 这里的-表示越远离阈值，概率越低

    # 是否打印过d<=0
    printed_flag = False
    # 打印每个点的距离、概率和真实标签
    for i, (distance, prob, true_label) in enumerate(zip(distances, probabilities, true_labels)):
        if distance <= 0 & printed_flag == False:
            print("********此时d<=0********")
            printed_flag = True
        print(
            f"Point {i + 1}: Distance to Threshold = {distance:.4f}, Probability of Normal data = {prob:.4f}, "
            f"Probability of Artifact data = {1 - prob:.4f}, True Label = {true_label}")
        visualize_sigmoid_probabilities(distance, prob, true_label)
    plt.show()
    # plt.legend()
    # 可视化阈值和Sigmoid概率
    # visualize_sigmoid_probabilities(distances, probabilities)


def calculate_confusion_matrix_scores(y_true_label, predict_label):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true_label, predict_label, labels=[1, 0])
    cm[1, 0], cm[0, 1] = cm[0, 1], cm[1, 0]
    # 计算真正例率（召回率）
    recall = recall_score(y_true_label, predict_label, pos_label=1)

    # 计算假正例率
    fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1])

    # 计算精准率
    precision = precision_score(y_true_label, predict_label, pos_label=1)

    # 计算 F1 分数
    f1 = f1_score(y_true_label, predict_label, pos_label=1)

    return recall, fpr, precision, f1, cm


def quality_assessment(good_data_dir, bad_data_dir, artifact_type):
    # ##################################################
    good_dataset = Val_dHCP_DwiData(good_data_dir, 'good')
    # 这里输入ghost
    ghost_dataset = Test_dHCP_DwiData(bad_data_dir, artifact_type)
    # 虽然加载了一个/多个sub的全部volume都含有伪影的切片，但是我们需要模拟的是现实生活中出现的少量伪影的情况,因此
    # 需要对加载的所有切片进行随机抽取部分数量的切片,形成一个新的dataloader,模拟一个subject中不可能每个volume都有伪影
    # 伪影可能是跳跃出现的,比如[:,:,:, 9、11、13、28、79]出现的伪影
    # 假设 data_loader 是已经实例化好的 DataLoader 对象
    original_good_dataset = good_dataset
    subsampled_indices = random.sample(range(len(original_good_dataset)), num_good_slicer)
    index_list = list(range(len(original_good_dataset) // 10))  # 获取一半数量的索引

    subsampled_good_dataset = Subset(original_good_dataset, subsampled_indices)

    original_dataset = ghost_dataset
    subsampled_indices = random.sample(range(len(original_dataset)), num_slicer)
    index_list = list(range(len(original_dataset) // 10))  # 获取一半数量的索引
    print("######索引#########")
    print(index_list)
    print("####新索引###########")
    print(subsampled_indices)
    # 创建子采样后的 Dataset
    subsampled_dataset = Subset(original_dataset, subsampled_indices)

    # 使用子采样后的 Dataset 创建新的 DataLoader
    val_good_dataloader = DataLoader(subsampled_good_dataset, batch_size=val_batch_size, shuffle=True)
    val_ghost_dataloader = DataLoader(subsampled_dataset, batch_size=val_batch_size, shuffle=True)
    # val_ghost_dataloader = DataLoader(ghost_dataset, batch_size=val_batch_size, shuffle=True)
    print("验证集DataLoader加载完毕!")

    # #后面假如一个sub样本###
    print("测试集DataLoader加载完毕!")
    # 前向传播好的数据
    # ####################验证 获取好的与坏的MSE、SSIM列表################
    val_good_MSE_loss, val_good_SSIM_loss, val_good_PSNR_loss = Validate(val_good_dataloader)
    val_ghost_MSE_loss, val_ghost_SSIM_loss, val_ghost_PSNR_loss = Validate(val_ghost_dataloader)

    # 在使用mu-2sigma之前，需要检验数据是否符合正态分布
    # ###########KS检验
    # check_normality_with_KS_test(val_good_MSE_loss, val_ghost_MSE_loss, 'val_good_MSE_loss')
    # check_normality_with_KS_test(val_good_SSIM_loss, val_ghost_SSIM_loss, 'val_good_SSIM_loss')
    # check_normality_with_KS_test(val_good_PSNR_loss, val_ghost_PSNR_loss, 'val_good_PSNR_loss')

    # good与ghost mse和ssim比较
    # good_vs_artifact_ssim_scatter_plot(val_good_SSIM_loss, val_ghost_SSIM_loss, 'Ghost')
    # ssim_mean_variance_dual_line_classification(val_good_SSIM_loss, val_ghost_SSIM_loss, 'Ghost')
    # 这里是Ghost
    plot_two_data_shuffle_divide_with_threshold(val_good_SSIM_loss, val_ghost_SSIM_loss, artifact_type)

    # 检验该批样本是否符合正态分布,由于切比雪夫不等式,样本量大的情况下可以不进行此步骤
    # is_normality_with_KS_test(val_good_SSIM_loss, val_ghost_SSIM_loss)
    threshold_ghost, accuracy_ghost, recall_ghost, fpr_ghost, precision_ghost, f1_ghost, cm_ghost = evaluate_unsupervised_performance(
        val_good_SSIM_loss, val_ghost_SSIM_loss)
    calculate_distance_and_probabilities_2(threshold_ghost, val_good_SSIM_loss, val_ghost_SSIM_loss)

    print("***************" * 3)
    print("该批样本测试结束，具体结果如下：")
    print("共检测样本数目：", len(val_good_SSIM_loss + val_ghost_SSIM_loss))
    print("含有标准样本：", len(val_good_SSIM_loss))
    print("含有伪影样本：", len(val_ghost_SSIM_loss))
    print(f"模型识别好数据和坏数据的正确率: {accuracy_ghost * 100:.3f}%")
    print(f"召回率Recall: {recall_ghost * 100:.3f}%")
    print(f"F1得分: {f1_ghost * 100:.3f}%")
    print("打印出混淆矩阵：")
    print(cm_ghost)
    return threshold_ghost


def test_single_subject(threshold, test_dir, test_type):
    if test_type == 'good':
        dataset = Test_Single_Subject_DwiData(test_dir, 'good')
    else:
        dataset = Test_Single_Subject_DwiData(test_dir, test_type)
    print("此次测试的subject是:" + test_type)
    test_single_dataloader = DataLoader(dataset, batch_size=val_batch_size, shuffle=True)
    # 第二个参数返回SSIM，第一个是MSE，第三个PSNR
    _, single_subject_ssim_list, _ = Test(test_single_dataloader)

    # 创建标签，如果原本就是好数据,标签为0；如果原本是坏数据，标签为1
    if test_type == 'good':
        true_labels = [0] * len(single_subject_ssim_list)
    else:
        true_labels = [1] * len(single_subject_ssim_list)

    # 将ssim点和阈值可视化
    plt.scatter(range(len(single_subject_ssim_list)), single_subject_ssim_list, label='SSIM')
    # 添加阈值水平线
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    # 设置图例
    plt.legend()
    # 显示图形
    if test_type == 'good':
        plt.title(f"Only Standard Data with Random Subject")
    else:
        plt.title(f"Only Artifact {test_type} Data with Random Subject")
    plt.show()

    # 用阈值进行预测 single_subject_ssim_list中大于阈值的认为是好 小于的认为是坏
    predict_labels = [0 if x > threshold else 1 for x in single_subject_ssim_list]
    # 打印出准确率
    accuracy = np.sum(np.array(predict_labels) == np.array(true_labels)) / len(true_labels)

    recall, fpr, precision, f1, cm = calculate_confusion_matrix_scores(true_labels, predict_labels)

    print("###############随机挑选的一个Subject####################3")
    print("该批样本测试结束，具体结果如下：")
    print("共检测样本切片数目：", len(single_subject_ssim_list))
    if test_type == 'good':
        print("含有标准样本：", len(single_subject_ssim_list))
        print("含有伪影样本：", 0)
    else:
        print("含有标准样本：", 0)
        print("含有伪影样本：", len(single_subject_ssim_list))
    print(f"模型识别好数据和坏数据的正确率: {accuracy * 100:.3f}%")
    print(f"召回率Recall: {recall * 100:.3f}%")
    # print(f"F1得分: {f1 * 100:.3f}%")
    print("根据上述实验选取的阈值得到混淆矩阵：")
    print(cm)


if __name__ == "__main__":
    # 测试、验证dataset路径和load模型路径
    val_dwi_dir = ''  # good dataset
    test_dwi_dir = ''  # with  artifact dataset

    # test_single_dir = '/media/overwater/E/NIMG_data/Final_Caffine/Test_single_subject'
    model_dir = 'checkpoint/best_model.pth'
    # ############加载模型########################################
    model = VectorQuantizedVAE(num_channels, hidden_size)
    model.load_state_dict(torch.load(model_dir, map_location='cpu'))
    # 超参数设置
    """
    lamda.
        Indicate whether we use mu-lamda*sigma
        Whether to use the 2sigma principle or the 3sigma principle
    k_sigmoid.
        Indicates how much we want to change the sigmoid tilt, i.e. the slope.
        The larger the k_sigmoid, the earlier the sigmoid function converges.
    val_batch_size.
        For better visualisation and ease of calculation of statistics for individual samples, the
        We set the batch_size to 1 for the test phase.
    """
    lamda = 2
    k_sigmoid = 20
    MSE = nn.MSELoss(reduction='mean')
    val_batch_size = 1
    # Number of randomly sampled data from all data in dataset
    num_good_slicer = 1000
    num_slicer = 100
    # 模型测试：
    model.eval()
    with torch.no_grad():
        threshold_ghost = quality_assessment(val_dwi_dir, test_dwi_dir, 'ghost')
        threshold_spike = quality_assessment(val_dwi_dir, test_dwi_dir, 'spike')
        threshold_noise = quality_assessment(val_dwi_dir, test_dwi_dir, 'noise')
        threshold_swap = quality_assessment(val_dwi_dir, test_dwi_dir, 'swap')
