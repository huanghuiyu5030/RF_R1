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

# 用于存储用户输入的值
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
        # ===== 修改：分类特征使用 labels 显示友好文本 =====
        options = properties["options"]
        labels = properties["labels"]
        default_val = properties["default"]
        # 找到默认值的索引
        try:
            default_index = options.index(default_val)
        except ValueError:
            default_index = 0
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=options,
            format_func=lambda x, labels=labels: labels[x],  # 关键：用 labels 显示
            index=default_index
        )
    feature_values.append(value)

if st.button("Predict"):
    # ===== 修改：创建带特征名的 DataFrame =====
    feature_names = list(feature_ranges.keys())
    input_df = pd.DataFrame([feature_values], columns=feature_names)

    # 模型预测
    predicted_class = model.predict(input_df)[0]          # 使用 DataFrame
    predicted_proba = model.predict_proba(input_df)[0]

    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果（使用通用衬线字体避免字体缺失错误）
    text = f"Based on feature values, predicted possibility of Thoracolumbar_fractures_shell is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='serif',           # ===== 修改：'Times New Roman' -> 'serif' =====
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)        # 使用 DataFrame

    # 生成 SHAP 力图（使用 matplotlib 模式返回 figure 对象）
    class_index = predicted_class
    # ===== 修改：正确保存 SHAP 图 =====
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:, :, class_index],
        input_df,
        matplotlib=True,            # 返回 matplotlib figure
        show=False                   # 不立即显示
    )
    # 保存 figure 到文件
    shap_fig.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")