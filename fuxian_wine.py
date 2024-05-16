import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_text
import FIRE

# 读取数据
red_wine = pd.read_csv('./data/wine/winequality-red.csv', sep=';')

# 分割特征和标签
X = red_wine.drop('quality', axis=1)
y = red_wine['quality']

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
mae_full = mean_absolute_error(y_test, y_pred)
r2_full = r2_score(y_test, y_pred)
pearson_full= pearsonr(y_test, y_pred)[0]
print(f"Full model MSE: {mse_full}")
print(f"Full model MAE: {mae_full}")
print(f"Full model R2: {r2_full}")
print(f"Full model pearson_high: {pearson_full}")

# 生成稀疏矩阵
tree_list = [tree for tree in rf.estimators_]
M_train, nodes_train = FIRE.get_node_harvest_matrix_sparse(pd.DataFrame(X_train), tree_list)
M_test, nodes_test = FIRE.get_node_harvest_matrix_sparse(pd.DataFrame(X_test), tree_list)

# 提取规则，低融合惩罚
gamma = 1.1
lambda_s = 0.01
lambda_f_low = 0.5 * lambda_s
breakpoints = []  # 确保breakpoints是一个空列表

w_low, iter_low, loss_sequence_low = FIRE.MCP_fuse_GBCD(
    M=M_train, y=y_train, blocks=[0, M_train.shape[1]],
    alpha_lasso=lambda_s, gamma=gamma, alpha_fuse=lambda_f_low,
    breakpoints=breakpoints, num_prox_iters=10, threshold=1e-4
)

# 评估低融合惩罚下的模型
y_pred_low = M_test @ w_low
mse_low = mean_squared_error(y_test, y_pred_low)
mae_low = mean_absolute_error(y_test, y_pred_low)
r2_low = r2_score(y_test, y_pred_low)
pearson_low = pearsonr(y_test, y_pred_low)[0]
print(f"Low λ_f model MSE: {mse_low}")
print(f"Low λ_f model MAE: {mae_low}")
print(f"Low λ_f model R2: {r2_low}")
print(f"Low λ_f model pearson_high: {pearson_low}")

# 提取规则，高融合惩罚
lambda_f_high = 2 * lambda_s

w_high, iter_high, loss_sequence_high = FIRE.MCP_fuse_GBCD(
    M=M_train, y=y_train, blocks=[0, M_train.shape[1]],
    alpha_lasso=lambda_s, gamma=gamma, alpha_fuse=lambda_f_high,
    breakpoints=breakpoints, num_prox_iters=10, threshold=1e-4
)

# 评估高融合惩罚下的模型
y_pred_high = M_test @ w_high
mse_high = mean_squared_error(y_test, y_pred_high)
mae_high = mean_absolute_error(y_test, y_pred_high)
r2_high = r2_score(y_test, y_pred_high)
pearson_high = pearsonr(y_test, y_pred_high)[0]
print(f"High λ_f model MSE: {mse_high}")
print(f"High λ_f model MAE: {mae_high}")
print(f"High λ_f model R2: {r2_high}")
print(f"High λ_f model pearson_high: {pearson_high}")

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
