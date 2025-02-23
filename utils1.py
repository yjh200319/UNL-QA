import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import numpy as np


# 一般数据量少时用S-W检验(即W检验) 例如数据<=50
# 但数据量大时，一般>50(chatgpt说>30) 进行K-S检验
def check_normality_with_SW_test(data1, data2, data_type):
    # 生成或者准备一个数据列表
    # data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # 将好的和坏的数据合并看是否符合正态分布

    # 先进行Z-score标准化
    all_data = data1 + data2
    z_score_data = (all_data - np.mean(all_data)) / np.std(all_data)

    # 进行Shapiro-Wilk检验
    statistic, p_value = stats.shapiro(all_data)

    print("*******" * 3 + data_type + "*******" * 3)
    # 打印检验结果
    print("Shapiro-Wilk检验统计量为：" + str(statistic))
    print("p值为：" + str(p_value))

    # 根据p值判断是否符合正态分布
    alpha = 0.05
    if p_value > alpha:
        print(f"{data_type} 数据可能符合正态分布")
    else:
        print(f"{data_type} 数据不符合正态分布")


def check_normality_with_KS_test(data1, data2, data_type):
    """
        使用K-S检验检验数据是否符合正态分布
        Args:
        data: 一个代表数据的列表或数组
        Returns:
        boolean: True表示数据可能符合正态分布，False表示数据不符合正态分布
    """
    # 第一种Z-score分别Z-score 错误！会把坏数据变成好数据
    # new_data1 = (data1 - np.mean(data1)) / np.std(data1)
    # new_data2 = (data2 - np.mean(data2)) / np.std(data2)
    # plt.scatter(range(len(new_data1)), new_data1, label='Good')
    # plt.scatter(range(len(new_data2)), new_data2, label=f'{data_type} Artifact')
    # plt.legend()
    # plt.title("After Separate Z-score")
    # plt.show()

    # 第二种z-score 一起z-score
    # all_data1 = data1 + data2
    # new_data2 = (all_data1 - np.mean(all_data1)) / np.std(all_data1)
    #
    # plt.scatter(range(len(data1)), new_data2[:len(data1)], label='Good', )
    # plt.scatter(range(len(data2)), new_data2[-len(data2):], label=f'{data_type} Artifact')
    # plt.legend()
    # plt.title("After All Z-score")
    # plt.show()

    # 生成或者准备一个数据列表
    # data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # 先进行Z-score标准化
    all_data = data1 + data2
    # z_score_data = (all_data - np.mean(all_data)) / np.std(all_data)

    # all_data = data1 + data2
    # 进行Shapiro-Wilk检验
    # statistic, p_value = stats.kstest(new_data2, 'norm', (np.mean(new_data2), np.std(new_data2)))
    statistic, p_value = stats.kstest(all_data, 'norm', (np.mean(all_data), np.std(all_data)))
    print("*******" * 3 + data_type + "*******" * 3)
    # 打印检验结果
    print("K-S检验统计量为：" + str(statistic))
    print("p值为：" + str(p_value))

    # 根据p值判断是否符合正态分布
    alpha = 0.05
    if p_value > alpha:
        print(f"{data_type} 数据符合正态分布")
    else:
        print(f"{data_type} 数据不符合正态分布")

    # 示例数据
    # 绘制直方图
    plt.hist(all_data, bins=8, color='skyblue', edgecolor='black')  # 指定直方图的颜色和边缘颜色
    plt.xlabel(f'{data_type} Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Data')
    plt.show()


def mse_mean_variance_dual_line_classification(good_data, bad_data, artifact_type):
    # 假设你已经有了两批数据 good_data 和 bad_data，它们都是一维的列表

    # 合并两批数据
    all_data = good_data + bad_data

    # 计算整体均值和标准差
    overall_mean = np.mean(all_data)
    overall_std = np.std(all_data)

    # 只计算good的均值和标准差
    good_mean = np.mean(good_data)
    good_std = np.std(good_data)

    # 生成散点图
    plt.scatter(range(len(good_data)), good_data, label='Good')
    plt.scatter(range(len(bad_data)), bad_data, label=f'{artifact_type} Artifact')

    # 画出虚线
    # plt.axhline(y=overall_mean + 2 * np.sqrt(overall_var), color='r', linestyle='--', label='Overall Mean+2*Var')
    # plt.axhline(y=overall_mean - 2 * np.sqrt(overall_var), color='r', linestyle='--', label='Overall Mean-2*Var')

    plt.axhline(y=overall_mean + 1 * overall_std, color='k', linestyle='--', label=r'$\mu+\sigma$')
    plt.axhline(y=overall_mean - 1 * overall_std, color='g', linestyle='--', label=r'$\mu-\sigma$')
    plt.axhline(y=overall_mean - 2 * overall_std, color='m', linestyle='--', label=r'$\mu-2\sigma$')
    plt.axhline(y=overall_mean + 2 * overall_std, color='y', linestyle='--', label=r'$\mu+2\sigma$')

    plt.axhline(y=good_mean + 2 * good_std, color='brown', linestyle='--', label=r'Only Good $\mu+2\sigma$')
    plt.axhline(y=good_mean - 2 * good_std, color='teal', linestyle='--', label=r'Only Good $\mu-2\sigma$')

    plt.axhline(y=overall_mean, color='r', linestyle='--', label=r'$\mu$')
    plt.xlabel("Slice Num")
    plt.ylabel("MSE")
    plt.title(f'Comparison of Good MSE and Artifact {artifact_type} MSE')
    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()


def ssim_mean_variance_dual_line_classification(good_data, bad_data, artifact_type):
    # 假设你已经有了两批数据 good_data 和 bad_data，它们都是一维的列表

    # 合并两批数据
    all_data = good_data + bad_data

    # 计算整体均值和标准差
    overall_mean = np.mean(all_data)
    overall_std = np.std(all_data)

    # 只计算good的均值和标准差
    good_mean = np.mean(good_data)
    good_std = np.std(good_data)

    # 生成散点图
    plt.scatter(range(len(good_data)), good_data, label='Good')
    plt.scatter(range(len(bad_data)), bad_data, label=f'{artifact_type} Artifact')

    # 画出虚线
    # plt.axhline(y=overall_mean + 2 * np.sqrt(overall_var), color='r', linestyle='--', label='Overall Mean+2*Var')
    # plt.axhline(y=overall_mean - 2 * np.sqrt(overall_var), color='r', linestyle='--', label='Overall Mean-2*Var')

    plt.axhline(y=overall_mean + 1 * overall_std, color='k', linestyle='--', label=r'$\mu+\sigma$')
    plt.axhline(y=overall_mean - 1 * overall_std, color='g', linestyle='--', label=r'$\mu-\sigma$')
    plt.axhline(y=overall_mean - 2 * overall_std, color='m', linestyle='--', label=r'$\mu-2\sigma$')
    plt.axhline(y=overall_mean + 2 * overall_std, color='y', linestyle='--', label=r'$\mu+2\sigma$')

    plt.axhline(y=good_mean + 2 * good_std, color='brown', linestyle='--', label=r'Only Good $\mu+2\sigma$')
    plt.axhline(y=good_mean - 2 * good_std, color='teal', linestyle='--', label=r'Only Good $\mu-2\sigma$')

    plt.axhline(y=overall_mean, color='r', linestyle='--', label=r'$\mu$')
    plt.xlabel("Slice Num")
    plt.ylabel("SSIM")
    plt.title(f'Comparison of Good SSIM and Artifact {artifact_type} SSIM')
    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()


def psnr_mean_variance_dual_line_classification(good_data, bad_data, artifact_type):
    # 假设你已经有了两批数据 good_data 和 bad_data，它们都是一维的列表

    # 合并两批数据
    all_data = good_data + bad_data

    # 计算整体均值和标准差
    overall_mean = np.mean(all_data)
    overall_std = np.std(all_data)
    # 只计算good的均值和标准差
    good_mean = np.mean(good_data)
    good_std = np.std(good_data)

    # 生成散点图
    plt.scatter(range(len(good_data)), good_data, label='Good')
    plt.scatter(range(len(bad_data)), bad_data, label=f'{artifact_type} Artifact')

    # 画出虚线
    # plt.axhline(y=overall_mean + 2 * np.sqrt(overall_var), color='r', linestyle='--', label='Overall Mean+2*Var')
    # plt.axhline(y=overall_mean - 2 * np.sqrt(overall_var), color='r', linestyle='--', label='Overall Mean-2*Var')

    plt.axhline(y=overall_mean + 1 * overall_std, color='k', linestyle='--', label=r'$\mu+\sigma$')
    plt.axhline(y=overall_mean - 1 * overall_std, color='g', linestyle='--', label=r'$\mu-\sigma$')
    plt.axhline(y=overall_mean - 2 * overall_std, color='m', linestyle='--', label=r'$\mu-2\sigma$')
    plt.axhline(y=overall_mean + 2 * overall_std, color='y', linestyle='--', label=r'$\mu+2\sigma$')

    plt.axhline(y=good_mean + 2 * good_std, color='brown', linestyle='--', label=r'Only Good $\mu+2\sigma$')
    plt.axhline(y=good_mean - 2 * good_std, color='teal', linestyle='--', label=r'Only Good $\mu-2\sigma$')

    plt.axhline(y=overall_mean, color='r', linestyle='--', label=r'$\mu$')
    plt.xlabel("Slice Num")
    plt.ylabel("PSNR")
    plt.title(f'Comparison of Good PSNR and Artifact {artifact_type} PSNR')
    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()


def apply_train_svm(good_data_mse, good_data_ssim, bad_data_mse, bad_data_ssim, artifact_type):
    # 训练数据
    # good_data_mse = [0.1, 0.3, 0.2, 0.4, 0.25]
    # good_data_ssim = [0.8, 0.7, 0.75, 0.65, 0.9]
    # bad_data_mse = [0.6, 0.65, 0.55, 0.72, 0.68]
    # bad_data_ssim = [0.3, 0.35, 0.25, 0.4, 0.28]

    X_train = np.vstack((np.column_stack((good_data_mse, good_data_ssim)),
                         np.column_stack((bad_data_mse, bad_data_ssim))))
    y_train = np.array([1] * len(good_data_mse) + [-1] * len(bad_data_mse))

    # 创建SVM分类器并进行训练
    clf = svm.SVC(kernel='linear', C=10)
    clf.fit(X_train, y_train)

    # 可视化训练数据和决策边界
    # plt.scatter(good_data_mse, good_data_ssim, color='lightgreen', label='Good Training Data')
    # plt.scatter(bad_data_mse, bad_data_ssim, color='pink', label=f'{artifact_type} Training Data')

    plt.scatter(good_data_mse, good_data_ssim, label='Standard Data')
    plt.scatter(bad_data_mse, bad_data_ssim, label=f'{artifact_type} Data')
    # plt.legend()
    # plt.title("MSE combined with SSIM")
    # plt.show()
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])

    train_predictions = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)

    print(f"Training Set Accuracy:{train_accuracy * 100:.3f}%")
    # plt.legend([f'Train Accuracy: {train_accuracy:.3f}%)'], loc='lower right')
    plt.text(0.5, 0.5, f'Train Accuracy: {train_accuracy:.3f}%', ha='center', va='center')

    plt.xlabel('MSE')
    plt.ylabel('SSIM')
    # plt.legend()
    plt.legend([f'Standard Data', f'{artifact_type} Data', f'Train Accuracy: {train_accuracy:.3f}%)'])
    plt.title('Training Data with SVM Decision Boundary')
    plt.show()

    return clf


def apply_test_svm(svm_model, good_data_mse, good_data_ssim, bad_data_mse, bad_data_ssim, artifact_type):
    X_test = np.vstack((np.column_stack((good_data_mse, good_data_ssim)),
                        np.column_stack((bad_data_mse, bad_data_ssim))))
    # 测试集样本真实标签
    y_test = np.array([1] * len(good_data_mse) + [-1] * len(bad_data_mse))

    plt.scatter(good_data_mse, good_data_ssim, label='True Standard Data')
    plt.scatter(bad_data_mse, bad_data_ssim, label=f'True {artifact_type} Data')
    # plt.legend()
    # 使用返回的模型进行预测
    # 测试集样本预测标签
    y_predictions = svm_model.predict(X_test)
    # print("Predictions:", y_predictions)
    test_accuracy = accuracy_score(y_test, y_predictions)

    print(f"Testing Set Accuracy:{test_accuracy * 100:.3f}%")
    # plt.legend([f'Testing Accuracy: {test_accuracy:.3f}%)'], loc='lower right')

    plt.legend([f'True Standard Data', f'True {artifact_type} Data', f'Test Accuracy: {test_accuracy:.3f}%)'])

    plt.text(0.5, 0.5, f'Test Accuracy: {test_accuracy:.3f}%', ha='center', va='center')

    # 可视化训练数据和决策边界
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired, marker='o', s=50)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.xlabel('MSE')
    plt.ylabel('SSIM')
    plt.title('SVM Decision Boundary with Testing Data')
    plt.show()


