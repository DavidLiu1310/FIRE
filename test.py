import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import FIRE

# 读取数据
red_wine = pd.read_csv('./data/wine/winequality-red.csv', sep=';')

# 分割特征和标签
X = red_wine.drop('quality', axis=1)
y = red_wine['quality']

features = X.columns

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 生成稀疏矩阵
tree_list = [DecisionTreeRegressor().fit(X_train, y_train) for _ in range(10)]
M, nodes = FIRE.get_node_harvest_matrix_sparse(pd.DataFrame(X_train), tree_list)

# 选择一个模型来训练，这里以l1_GBCD为例
w, iter1, loss_sequence = FIRE.MCP_GBCD(M=M, y=y_train, blocks=[0, M.shape[1]], alpha_lasso=0.01, num_prox_iters=10,
                                       threshold=1e-4)

# 查看训练后的权重和损失变化
print("模型参数:", w)
print("迭代次数:", iter1)
print("损失序列:", loss_sequence)

import matplotlib.pyplot as plt


# 1. 模型性能评估
def evaluate_model(M, w, X_test, y_test):
    M_test, _ = FIRE.get_node_harvest_matrix_sparse(pd.DataFrame(X_test), tree_list)
    y_pred = M_test @ w
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pearson = pearsonr(y_test, y_pred)[0]
    return mse, mae, r2, pearson


# 评估模型在测试集上的表现
mse, mae, r2, pearson = evaluate_model(M, w, X_test, y_test)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")
print(f"pearson: {pearson}")

# 2. 模型参数解释
non_zero_weights = np.sum(w != 0)
print(f"Number of non-zero weights: {non_zero_weights}")
print(f"Total number of weights: {len(w)}")

# 3. 可视化损失序列
plt.plot(loss_sequence)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Sequence')
plt.show()

import numpy as np
import pandas as pd
from sklearn.tree import export_text


# 假设 tree_list 是训练模型中使用的决策树列表
def extract_rules(tree_list, w):
    rules = []
    for i, tree in enumerate(tree_list):
        if w[i] != 0:
            tree_rules = export_text(tree)
            rules.append((w[i], tree_rules))
    return rules


# 获取非零权重对应的规则
rules = extract_rules(tree_list, w)

# 打印规则
for weight, rule in rules:
    print(f"Rule weight: {weight}")
    print(rule)
    print("----------")

# from sklearn.tree import export_graphviz
# import graphviz
#
#
# # 可视化规则
# def visualize_tree(tree, feature_names):
#     dot_data = export_graphviz(tree, out_file=None, feature_names=feature_names, filled=True)
#     return graphviz.Source(dot_data)
#
#
# features = red_wine.columns[:-1]
# for i, tree in enumerate(tree_list):
#     if w[i] != 0:
#         print(f"Visualizing tree with weight: {w[i]}")
#         tree_viz = visualize_tree(tree, feature_names=features)
#         tree_viz.render(view=True, format='png')  # 改为 'png' 格式


'''
强化学习
搜索空间
优化问题

强化学习 + 规则 + 表格
'''