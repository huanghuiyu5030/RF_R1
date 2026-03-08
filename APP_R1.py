import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('rf.pkl')

# 特征范围定义
feature_ranges = {
    "DVC": {"type": "numerical", "min": 0.000, "max": 1.000, "default": 0.89},
    "BMI": {"type": "numerical", "min": 10.000, "max": 50.000, "default": 28.555},
    "Age": {"type": "numerical", "min": 0.000, "max": 100.000, "default": 56},
    "DVR": {"type": "numerical", "min": 0.000, "max": 1.000, "default": 0.93},
    "Cobb": {"type": "numerical", "min": 0, "max": 50, "default": 34},
    "AO_Spine": {"type": "categorical", "options": [0,1,2,3], "default": 2, "labels": {0: "A1", 1: "A2", 2: "A3", 3: "A4"}},
    "BMD": {"type": "categorical", "options": [0,1,2], "default": 2, "labels": {0: "T ≥ -1.0", 1: "-2.5 <T <-1.0", 2:"T ≤ -2.5"}},
    "Gender": {"type": "categorical", "options": [0,1], "default": 0, "labels": {0: "Female", 1: "Male"}},
}

st.title("Prediction Model with SHAP Visualization")

st.header("Enter the following feature values:")

feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        options = properties["options"]
        labels = properties["labels"]
        default_val = properties["default"]
        try:
            default_index = options.index(default_val)
        except ValueError:
            default_index = 0
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=options,
            format_func=lambda x, labels=labels: labels[x],
            index=default_index
        )
    feature_values.append(value)

if st.button("Predict"):
    # 创建带特征名的 DataFrame，并确保列顺序与模型一致
    feature_names = list(feature_ranges.keys())
    input_df = pd.DataFrame([feature_values], columns=feature_names)

    # 如果模型保存了特征名，则按模型顺序重新排列列
    if hasattr(model, 'feature_names_in_'):
        try:
            input_df = input_df[model.feature_names_in_]
        except KeyError as e:
            st.error(f"输入特征与模型不匹配。期望的特征名: {list(model.feature_names_in_)}")
            st.stop()

    # 模型预测
    predicted_class = model.predict(input_df)[0]
    predicted_proba = model.predict_proba(input_df)[0]

    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果
    text = f"Based on feature values, predicted possibility of ISP is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='DejaVu Serif',  # 通用衬线字体
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # 处理 shap_values 格式（兼容二分类和多分类）
    if isinstance(shap_values, list):
        # 二分类：shap_values 是长度为2的列表
        shap_vals = shap_values[predicted_class]
        expected_value = explainer.expected_value[predicted_class]
    else:
        # 多分类：三维数组 (样本数, 特征数, 类别数)
        shap_vals = shap_values[:, :, predicted_class]
        expected_value = explainer.expected_value[predicted_class] if isinstance(explainer.expected_value, list) else explainer.expected_value

    # ===== 修改：安全地将 expected_value 转换为标量 =====
    if hasattr(expected_value, 'size') and expected_value.size == 1:
        expected_value = expected_value.item()
    elif isinstance(expected_value, list) and len(expected_value) == 1:
        expected_value = expected_value[0]
    # 如果已经是标量（int/float），保持不变
    # ===================================================

    # 确保 shap_vals 是二维 (1, n_features)
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(1, -1)

    # 生成 SHAP 力图（使用 matplotlib 模式返回 figure 对象）
    shap_fig = shap.force_plot(
        expected_value,
        shap_vals,
        input_df,
        matplotlib=True,
        show=False
    )

    # 保存并显示 SHAP 图
    shap_fig.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")