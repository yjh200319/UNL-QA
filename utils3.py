import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest, norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # Sigmoid 函数

# 设定随机种子，可以是任意整数
random_seed = 2024030
random.seed(random_seed)


def plot_two_data(data1, data2):
    # 将A组和B组数据合并
    ssim_data = data1 + data2

    # 打乱数据顺序，使得A组和B组数据交叉
    random.shuffle(ssim_data)

    # 打印生成的数据
    print("A组数据:", data1)
    print("B组数据:", data2)
    print("合并后的数据:", ssim_data)

    plt.scatter(range(len(data1)), data1, label='Good data')
    plt.scatter(range(len(data2)), data2, label='Bad data')
    plt.title('Comparison of Good and Bad data')
    plt.xlabel('SSIM Value')
    plt.ylabel('Slice Num')
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
    plt.legend()
    plt.title(f'Comparison of Good and {artifact_type} data')
    plt.xlabel('Slice Num')
    plt.ylabel('SSIM Value')

    # 显示图表
    plt.show()


def is_normality_with_KS_test(data1, data2):
    ssim_data = data1 + data2
    # 执行K-S检验
    ks_statistic, ks_p_value = kstest(ssim_data, 'norm', (np.mean(ssim_data), np.std(ssim_data)))

    # 打印K-S检验结果
    print("K-S统计量:", ks_statistic)
    print("P值:", ks_p_value)
    alpha = 0.05
    if ks_p_value < alpha:
        print("检验不通过,不符合正态分布")
    else:
        print("检验通过,符合正态分布!")
    # 绘制合并后的数据和正态分布的累积分布函数（CDF）进行比较
    plt.hist(ssim_data, bins=20, density=True, alpha=0.7, label='Combined Data')
    plt.plot(sorted(ssim_data), norm.cdf(sorted(ssim_data)), 'r-', label='Normalization_CDF')
    plt.xlabel('SSIM Value')
    plt.ylabel('Frequency')
    # plt.title('合并后的数据和正态分布CDF比较')
    plt.title('Comparison of merged data and normally distributed CDFs')
    plt.legend()
    plt.show()


def is_normality_with_KS(data1):
    # 执行K-S检验
    ks_statistic, ks_p_value = kstest(data1, 'norm', (np.mean(data1), np.std(data1)))

    # 打印K-S检验结果
    print("K-S统计量:", ks_statistic)
    print("P值:", ks_p_value)
    alpha = 0.05
    if ks_p_value < alpha:
        print("检验不通过,不符合正态分布")
    else:
        print("检验通过,符合正态分布!")
    # 绘制合并后的数据和正态分布的累积分布函数（CDF）进行比较
    plt.hist(data1, bins=5, density=True, alpha=0.7, label='Combined Data')
    # plt.plot(sorted(data1), norm.cdf(sorted(data1)), 'r-', label='Normalization_CDF')
    plt.xlabel('SSIM Value')
    plt.ylabel('Frequency')
    # plt.title('合并后的数据和正态分布CDF比较')
    plt.title('Comparison of merged data and normally distributed CDFs')
    plt.legend()
    plt.show()


def evaluate_unsupervised_performance(ssim_A, ssim_B):
    # 计算均值和标准差
    total = ssim_A + ssim_B
    mean, std = np.mean(total), np.std(total)

    # 计算阈值
    threshold = mean - lamda * std

    plt.axhline(y=threshold, color='red', linestyle='--', label=rf'Threshold $\mu-{lamda}\sigma$')

    # 添加标签和图例
    plt.scatter(range(len(ssim_A)), ssim_A, label='Good data')
    plt.scatter(range(len(ssim_B)), ssim_B, label='Artifact data')
    plt.title('SSIM Distribution with Threshold')
    plt.xlabel('Slice Num')
    plt.ylabel('SSIM Values')
    plt.legend()
    plt.show()
    # 判定异常点
    predicted_labels = [1 if ssim > threshold else 0 for ssim in ssim_A + ssim_B]

    # 计算正确率
    true_labels = [1] * len(ssim_A) + [0] * len(ssim_B)
    accuracy = np.sum(np.array(predicted_labels) == np.array(true_labels)) / len(true_labels)

    print(f"阈值: {threshold}")
    print(f"根据该阈值划分正确率: {accuracy * 100:.2f}%")

    return threshold, accuracy


# def calculate_distance_and_probabilities(ssim_A, ssim_B):
#     # 计算均值和标准差
#     mean_A, std_A = np.mean(ssim_A), np.std(ssim_A)
#     mean_B, std_B = np.mean(ssim_B), np.std(ssim_B)
#
#     # 计算阈值
#     threshold = mean_A - 2 * std_A
#
#     # 计算每个点到阈值的距离
#     distances = np.abs(np.array(ssim_A + ssim_B) - threshold)
#
#     # 使用Sigmoid函数计算概率
#     probabilities = expit(-distances)  # 这里的-表示越远离阈值，概率越低
#
#     # 打印每个点的距离和概率
#     for i, distance in enumerate(distances):
#         print(f"Point {i + 1}: Distance to Threshold = {distance:.4f}, Probability = {probabilities[i]:.4f}")
#
#     # 可视化阈值和Sigmoid概率
#     visualize_sigmoid_probabilities(distances, probabilities)


def calculate_distance_and_probabilities(T, ssim_A, ssim_B, k_sigmoid):
    # 计算每个点到阈值的距离
    total_ssim = ssim_A + ssim_B
    distances = total_ssim - T
    distances = np.array(sorted(distances, reverse=True))

    true_labels = [1] * len(ssim_A) + [0] * len(ssim_B)
    # distances = np.argsort(distances)
    # 使用Sigmoid函数计算概率
    probabilities = expit(k_sigmoid * distances)  # 这里的-表示越远离阈值，概率越低

    # 打印每个点的距离和概率
    for i, distance in enumerate(distances):
        print(
            f"Point {i + 1}: Distance to Threshold = {distance:.4f}, Probability of Normal data = {probabilities[i]:.4f}, "
            f"Probability of Artifact data = {1 - probabilities[i]:.4f}")

    # 可视化阈值和Sigmoid概率
    # visualize_sigmoid_probabilities(distances, probabilities)


def calculate_distance_and_probabilities_2(T, ssim_A, ssim_B):
    # 计算每个点到阈值的距离
    total_ssim = ssim_A + ssim_B
    distances = total_ssim - T
    # distances = np.array(sorted(distances, reverse=True))

    true_labels = [1] * len(ssim_A) + [0] * len(ssim_B)
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


def visualize_sigmoid_probabilities(distances, probabilities, true_label):
    # 表明原本就是好数据
    if true_label == 1:
        plt.plot(distances, probabilities, marker='o', linestyle='-', color='orange')
    # 表明原本就是坏数据Artifact
    else:
        plt.plot(distances, probabilities, marker='o', linestyle='-', color='navy')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Decision Boundary (0.5)')
    plt.xlabel('Distance to Threshold')
    plt.ylabel('Probability')
    plt.title('Sigmoid Probability Based on Distance to Threshold')
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    # 生成A组数据（90个高SSIM值）
    a_data = [random.uniform(0.6, 0.8) for _ in range(200)]
    # 生成B组数据（10个低SSIM值）
    b_data = [random.uniform(0.4, 0.65) for _ in range(20)]
    # 选取\mu-k\sigma
    lamda = 2
    k_sigmoid = 10
    plot_two_data(a_data, b_data)
    plot_two_data_shuffle_divide_with_threshold(a_data, b_data)
    is_normality_with_KS_test(a_data, b_data)

    threshold_ghost, accuracy = evaluate_unsupervised_performance(a_data, b_data)
    calculate_distance_and_probabilities_2(threshold_ghost, a_data, b_data)
