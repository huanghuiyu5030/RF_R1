import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('rf.pkl')

# 特征范围定义（含友好标签）
feature_ranges = {
    "DVC": {"type": "numerical", "min": 0.000, "max": 1.000, "default": 0.89},
    "BMI": {"type": "numerical", "min": 10.000, "max": 50.000, "default": 28.555},
    "Age": {"type": "numerical", "min": 0.000, "max": 100.000, "default": 56},
    "DVR": {"type": "numerical", "min": 0.000, "max": 1.000, "default": 0.93},
    "Cobb": {"type": "numerical", "min": 0, "max": 50, "default": 34},
    "AO_Spine": {
        "type": "categorical",
        "options": [0, 1, 2, 3],
        "default": 2,
        "labels": {0: "A1", 1: "A2", 2: "A3", 3: "A4"}
    },
    "BMD": {
        "type": "categorical",
        "options": [0, 1, 2],
        "default": 2,
        "labels": {0: "T ≥ -1.0", 1: "-2.5 < T < -1.0", 2: "T ≤ -2.5"}
    },
    "Gender": {
        "type": "categorical",
        "options": [0, 1],
        "default": 0,
        "labels": {0: "Female", 1: "Male"}
    },
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
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
        # 按 options 顺序生成显示标签列表
        display_options = [properties["labels"][opt] for opt in properties["options"]]
        # 找到默认值在 options 中的索引
        default_index = properties["options"].index(properties["default"])
        selected_label = st.selectbox(
            label=f"{feature}",
            options=display_options,
            index=default_index,
        )
        # 将选中的标签转换回数值
        label_to_value = {properties["labels"][opt]: opt for opt in properties["options"]}
        value = label_to_value[selected_label]
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of Thoracolumbar_fractures_shell is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值（多分类模型返回列表，每个元素对应一个类别）
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图（针对预测类别）
    class_index = predicted_class  # 当前预测类别
    shap.force_plot(
        explainer.expected_value[class_index],          # 该类的期望值
        shap_values[class_index],                        # 该类的 SHAP 值（形状: 样本数×特征数）
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
        show=False                                        # 防止自动显示，便于保存
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")