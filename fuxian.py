import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_text

import FIRE

# 加载Census Planning Database数据
try:
    data = pd.read_csv('./data/pdb2019trv3_us.csv', encoding='ISO-8859-1')  # 确保文件路径正确
except UnicodeDecodeError:
    data = pd.read_csv('./data/pdb2019trv3_us.csv', encoding='latin1')  # 如果ISO-8859-1失败，尝试latin1

# 删除空缺值比例超过30%的列
threshold = 0.3
data = data.loc[:, data.isnull().mean() < threshold]

# 删除所有空缺值
data = data.dropna()
# 删除所有空缺值
data = data.dropna()
# 输出数据形状，确保删除空缺值后有足够的数据
print("Data shape after dropping NA:", data.shape)

# 检查是否删除了所有数据
if data.shape[0] == 0:
    raise ValueError("All data was removed after dropping NA values. Please check your dataset.")


# 选择数值类型的列
data = data.select_dtypes(include=[np.number])

# 分割特征和标签
X = data.drop('Self_Response_Rate_ACS_13_17', axis=1)
y = data['Self_Response_Rate_ACS_13_17']

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建随机森林模型
rf = RandomForestRegressor(n_estimators=500, max_depth=3, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse_full = mean_squared_error(y_test, y_pred)
print(f"Full model MSE: {mse_full}")

# 生成稀疏矩阵
tree_list = [tree for tree in rf.estimators_]
M_train, nodes_train = FIRE.get_node_harvest_matrix_sparse(pd.DataFrame(X_train), tree_list)
M_test, nodes_test = FIRE.get_node_harvest_matrix_sparse(pd.DataFrame(X_test), tree_list)

# 提取规则，低融合惩罚
gamma = 1.1
lambda_s = 0.01
lambda_f_low = 0.5 * lambda_s

w_low, iter_low, loss_sequence_low = FIRE.MCP_fuse_GBCD(
    M=M_train, y=y_train, blocks=[0, M_train.shape[1]],
    alpha_lasso=lambda_s, gamma=gamma, alpha_fuse=lambda_f_low,
    breakpoints=[], num_prox_iters=10, threshold=1e-4
)

# 评估低融合惩罚下的模型
y_pred_low = M_test @ w_low
mse_low = mean_squared_error(y_test, y_pred_low)
print(f"Low λ_f model MSE: {mse_low}")

# 提取规则，高融合惩罚
lambda_f_high = 2 * lambda_s

w_high, iter_high, loss_sequence_high = FIRE.MCP_fuse_GBCD(
    M=M_train, y=y_train, blocks=[0, M_train.shape[1]],
    alpha_lasso=lambda_s, gamma=gamma, alpha_fuse=lambda_f_high,
    breakpoints=[], num_prox_iters=10, threshold=1e-4
)

# 评估高融合惩罚下的模型
y_pred_high = M_test @ w_high
mse_high = mean_squared_error(y_test, y_pred_high)
print(f"High λ_f model MSE: {mse_high}")

# 提取并打印规则
def extract_rules(tree_list, w):
    rules = []
    for i, tree in enumerate(tree_list):
        if w[i] != 0:
            tree_rules = export_text(tree)
            rules.append((w[i], tree_rules))
    return rules

# 获取低融合惩罚下的规则
rules_low = extract_rules(tree_list, w_low)
print("Low λ_f rules:")
for weight, rule in rules_low:
    print(f"Rule weight: {weight}")
    print(rule)
    print("----------")

# 获取高融合惩罚下的规则
rules_high = extract_rules(tree_list, w_high)
print("High λ_f rules:")
for weight, rule in rules_high:
    print(f"Rule weight: {weight}")
    print(rule)
    print("----------")
