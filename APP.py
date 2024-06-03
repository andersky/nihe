import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 设置随机种子
seed = 50
np.random.seed(seed)
torch.manual_seed(seed)


# 定义数据转换函数
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


# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc6 = nn.Linear(50, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc6(x)
        return x


# 数据处理函数
def process_data(uploaded_file):
    data = pd.read_csv(uploaded_file, sep='\s+', header=None, names=['time', '1', 'correlation', '2'])
    data_t = data['time'].values
    data_y = data['correlation'].values

    t = convert_to_numbers(data_t)
    t = np.array(t, dtype=float)
    t = t[:-200]
    data_y = data_y[:-200]

    return t, data_y


# 绘图函数
def create_plots(t, data_y):
    t_tensor = torch.tensor(t, dtype=torch.float32).clone().detach().unsqueeze(1)
    y_tensor = torch.tensor(data_y, dtype=torch.float32).clone().detach().unsqueeze(1)

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

    # 设置字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    # 绘制真实曲线
    plt.figure(figsize=(10, 5))
    plt.plot(t, data_y, color='green', linestyle='--', label='True Curve')
    plt.xlabel('Times(fs)')
    plt.ylabel('Correlation')
    plt.title('Generated Data')
    plt.legend()
    true_curve = 'true_curve.png'
    plt.savefig(true_curve)
    plt.close()

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.title("Train Loss")
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    train_loss = 'train_loss.png'
    plt.savefig(train_loss)
    plt.close()

    # 绘制验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.title("Test Loss")
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    val_loss = 'val_loss.png'
    plt.savefig(val_loss)
    plt.close()

    # 绘制模型拟合曲线
    plt.figure(figsize=(10, 5))
    plt.title("Models Fit Curves")
    plt.scatter(t, data_y, s=10, label='Original Data')
    plt.plot(t, model(t_tensor).detach().numpy(), color='red', label='Fitted Curve')
    plt.xlabel('Times(fs)')
    plt.ylabel('Correlation')
    plt.legend()
    fit_curve = 'fit_curve.png'
    plt.savefig(fit_curve)
    plt.close()

    return true_curve, train_loss, val_loss, fit_curve


# 创建Streamlit界面
st.title('数据上传和分析')

st.write("你可以上传一个自己的 VISCOSITY.dat 文件，或者使用我们的示例文件进行测试。")

# 添加示例文件下载按钮
example_file_path = "Example_VISCOSITY.dat"
with open(example_file_path, "rb") as file:
    btn = st.download_button(
        label="下载示例文件",
        data=file,
        file_name="Example_VISCOSITY.dat",
        mime="text/plain"
    )

uploaded_file = st.file_uploader("上传VISCOSITY.dat文件", type="dat")

if uploaded_file is not None:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(
            """
            <div style="text-align: center;">
                <img src="https://th.bing.com/th/id/R.ea5977eff4ed49181ed417d141d853d1?rik=ZfV2wq9ZHgFh%2bA&riu=http%3a%2f%2fimages.shejidaren.com%2fwp-content%2fuploads%2f2019%2f04%2f084814cxl.jpg&ehk=Z10pn03Fw4ZYaYfqyojkPo91JgQkpnZhy5nuBTiiAUc%3d&risl=&pid=ImgRaw&r=0" width="400">
                <p>请稍等...</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.spinner("处理中，请稍等..."):
        t, data_y = process_data(uploaded_file)
        true_curve, train_loss, val_loss, fit_curve = create_plots(t, data_y)

    placeholder.empty()

    # 显示图片
    st.image(true_curve, caption='True Curve', use_column_width=True)
    st.image(train_loss, caption='Training Loss', use_column_width=True)
    st.image(val_loss, caption='Val Loss', use_column_width=True)
    st.image(fit_curve, caption='Models Fit Curves', use_column_width=True)
