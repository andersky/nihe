import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from matplotlib import font_manager as fm
import os

# 自动检测并设置中文字体
def get_zh_font():
    zh_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    for font in zh_fonts:
        if any(kw in font.lower() for kw in ['zh', 'chinese', 'hei', 'kai', 'fang']):
            return fm.FontProperties(fname=font)
    return None

zh_font = get_zh_font()
if zh_font is None:
    st.warning("未找到中文字体，使用默认字体。")

def process_data(file):
    data = pd.read_csv(file, sep='\s+', header=None, names=['time', '1', 'correlation', '2'])
    data_t = data['time'].values
    data_y = data['correlation'].values

    def convert_to_numbers(x):
        first_star_index = None
        for i, item in enumerate(x):
            if item == '*****':
                first_star_index = i
                break

        if first_star_index is None:
            return x

        value_for_star = float(x[first_star_index - 1]) + 0.5
        converted_x = []

        for i in range(first_star_index):
            converted_x.append(x[i])

        for i in range(first_star_index, len(x)):
            converted_x.append(value_for_star)
            value_for_star += 0.5

        return converted_x

    t = convert_to_numbers(data_t)
    t = np.array(t, dtype=float)
    t = t[:-200]
    data_y = data_y[:-200]

    return t, data_y

def create_plots(t, data_y):
    plt.figure(figsize=(10, 5))
    plt.plot(t, data_y, color='green', linestyle='--', label='真实曲线')
    plt.xlabel('时间', fontproperties=zh_font)
    plt.ylabel('相关性', fontproperties=zh_font)
    plt.title('生成数据', fontproperties=zh_font)
    plt.legend(prop=zh_font)
    true_curve_path = 'true_curve.png'
    plt.savefig(true_curve_path)
    plt.close()

    t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(data_y, dtype=torch.float32).unsqueeze(1)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(1, 1)
            self.fc2 = nn.Linear(1, 100)
            self.fc6 = nn.Linear(100, 1)
            self.dropout = nn.Dropout(0.5)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc6(x)
            return x

    model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    t_train, t_val, y_train, y_val = train_test_split(t_tensor, y_tensor, test_size=0.2, random_state=42)

    train_losses = []
    val_losses = []

    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(t_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(t_val)
            val_loss = criterion(val_output, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

    # 调整子图布局和大小
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.title("训练损失", fontproperties=zh_font)
    plt.plot(train_losses, label='训练损失')
    plt.xlabel('Epoch', fontproperties=zh_font)
    plt.ylabel('Loss', fontproperties=zh_font)
    plt.legend(prop=zh_font)
    train_loss_path = 'train_loss.png'
    plt.savefig(train_loss_path, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)  # 使用相同的子图索引
    plt.title("验证损失", fontproperties=zh_font)
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch', fontproperties=zh_font)
    plt.ylabel('Loss', fontproperties=zh_font)
    plt.legend(prop=zh_font)
    val_loss_path = 'val_loss.png'
    plt.savefig(val_loss_path, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title("模型拟合曲线", fontproperties=zh_font)
    plt.scatter(t, data_y, s=10, label='原始数据')
    plt.plot(t, model(t_tensor).detach().numpy(), color='red', label='拟合曲线')
    plt.xlabel('时间(fs)', fontproperties=zh_font)
    plt.ylabel('相关性', fontproperties=zh_font)
    plt.legend(prop=zh_font)
    fit_curve_path = 'fit_curve.png'
    plt.savefig(fit_curve_path)
    plt.close()

    return true_curve_path, train_loss_path, val_loss_path, fit_curve_path

# Streamlit应用界面
st.title('数据上传和分析')

st.write("你可以上传一个自己的 DIFFUSION.dat 文件，或者使用我们的示例文件进行测试。")

# 添加示例文件下载按钮
example_file_path = "Example_DIFFUSION.dat"
with open(example_file_path, "rb") as file:
    btn = st.download_button(
        label="下载示例文件",
        data=file,
        file_name="Example_DIFFUSION.dat",
        mime="text/plain"
    )

uploaded_file = st.file_uploader("上传一个 DIFFUSION.dat 文件")

if uploaded_file is not None:
    t, data_y = process_data(uploaded_file)
    true_curve, train_loss, val_loss, fit_curve = create_plots(t, data_y)

    # 使用streamlit的布局功能使图片居中显示
    st.image(true_curve, caption='真实曲线', use_column_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image(train_loss, caption='训练损失', use_column_width=True)
    with col2:
        st.image(val_loss, caption='验证损失', use_column_width=True)
    st.image(fit_curve, caption='模型拟合曲线', use_column_width=True)
