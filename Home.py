import streamlit as st
import json
import os

# 用户数据文件路径
USER_DATA_FILE = 'users_db.json'

# 读取用户数据
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as file:
            return json.load(file)
    return {}

# 保存用户数据
def save_user_data(users_db):
    with open(USER_DATA_FILE, 'w') as file:
        json.dump(users_db, file)

# 初始化用户数据库
if 'users_db' not in st.session_state:
    st.session_state.users_db = load_user_data()

def register(username, password):
    if username in st.session_state.users_db:
        return False, "用户已存在"
    else:
        st.session_state.users_db[username] = password
        save_user_data(st.session_state.users_db)
        return True, "注册成功"

def login(username, password):
    if username not in st.session_state.users_db:
        return False, "用户不存在"
    elif st.session_state.users_db[username] != password:
        return False, "密码错误"
    else:
        st.session_state.logged_in = True
        st.session_state.username = username
        return True, "登录成功"

def main():
    # 添加背景图片和自定义样式的CSS
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://img1.baidu.com/it/u=2362900004,1888179926&fm=253&fmt=auto&app=120&f=JPEG?w=1044&h=695");
            background-size: contain;
            background-position: center;
            background-repeat: no-repeat;
        }
        .custom-link {
            font-weight: bold; /* 字体粗细改为 bold */
            font-size: 24px; /* 调整“点击这里”的字体大小 */
            text-decoration: none;
            color: red; /* 链接文字颜色 */
        }
        .custom-link:hover {
            color: darkred; /* 鼠标悬停时的链接颜色 */
        }
        .success-message {
            color: green;
            font-size: 24px; /* 调整成功消息的字体大小 */
            font-weight: bold;
        }
        .welcome-message {
            font-size: 24px; /* 调整欢迎消息的字体大小 */
            font-weight: bold;
        }
        .input-label {
            font-size: 20px; /* 调整输入标签的字体大小 */
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("欢迎使用")

    menu = ["登录", "注册"]
    choice = st.sidebar.selectbox("选择操作", menu)

    if choice == "注册":
        st.subheader("注册")
        username = st.text_input("用户名", key="register_username", placeholder="用户名", help="请输入用户名")
        password = st.text_input("密码", type='password', key="register_password", placeholder="密码", help="请输入密码")
        confirm_password = st.text_input("确认密码", type='password', key="confirm_password", placeholder="确认密码", help="请再次输入密码")

        if st.button("注册"):
            if password == confirm_password:
                success, msg = register(username, password)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                st.error("密码不匹配")

    elif choice == "登录":
        st.subheader("登录")
        username = st.text_input("用户名", key="login_username", placeholder="用户名", help="请输入用户名")
        password = st.text_input("密码", type='password', key="login_password", placeholder="密码", help="请输入密码")

        if st.button("登录"):
            success, msg = login(username, password)
            # 如果登录成功
            if success:
                st.markdown(f'<div class="success-message">{msg}</div>', unsafe_allow_html=True)  # 显示成功消息
                st.markdown(f'<div class="welcome-message">欢迎, {username}!</div>', unsafe_allow_html=True)  # 显示欢迎消息
                st.balloons()  # 显示气球效果
                # 创建一个超链接，添加自定义样式
                st.markdown(
                    '<a href="https://7jsaxdnk2axrpxknnrqrrz.streamlit.app/" class="custom-link">请点击这里</a>',
                    unsafe_allow_html=True
                )
            else:
                st.error(msg)

    # Debug 信息
    # st.sidebar.subheader("Debug 信息")
    # st.sidebar.write("用户数据库:", st.session_state.users_db)

if __name__ == '__main__':
    main()
