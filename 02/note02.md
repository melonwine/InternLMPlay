# 轻松玩转书生·浦语大模型趣味演示

## 1. 大模型及InternLM模型简介

### 1.1 什么是大模型

    机器学习或人工智能领域中参数数量巨大、拥有庞大计算能力和参数规模的模型

### 1.2 特点及应用

- 利用大量数据进行训练

- 拥有数十亿甚至数千亿个参数

- 模型在各种任务中展现出惊人的性能

### 1.3 InternLM 模型全链条开源

    InternLM是一个开源的轻量级训练框架，旨在支持大模型训练而无需大量的依赖。基于InternLM训练框架，上海人工智能实验室已经发布了两个开源的预训练模型：InternLM-7B 和 InternLM-20B。

    Lagent 是一个轻量级、开源的基于大语言模型的智能体（agent）框架，支持用户快速地将一个大语言模型转变为多种类型的智能体。通过Lagent框架可以更好的发挥InternLM的全部性能。

     浦语·灵笔是基于书生·浦语大语言模型研发的视觉-语言大模型，提供出色的图文理解和创作能力，使用浦语·灵笔大模型可以轻松的创作一篇图文推文。

## 2. InternLM-Chat-7B智能对话

### 2.1 环境准备

    在 [InternStudio](https://studio.intern-ai.org.cn/) 平台中创建开发机，选择 A100(1/4) 的配置，镜像选择Cuda11.7-conda，启动后根据页面说明在本地终端通过SSH连接到开发机。依次执行以下命令：

```bash
# 执行bash进入conda环境
bash
# 从本地克隆一个已有的 pytorch 2.0.1 的环境
/root/share/install_conda_env_internlm_base.sh internlm-demo
# 激活环境
conda activate internlm-demo
# 升级pip
python -m pip install --upgrade pip
# 安装运行 demo 所需要的依赖
pip install modelscope==1.9.5
pip install transformers==4.35.2
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.24.1
```

### 2.2 模型下载

    [InternStudio](https://studio.intern-ai.org.cn/) 平台的share目录下已经准备了全系列的InternLM模型，直接用下面命令复制即可。

```
mkdir -p /root/model/ailab
cp -r /root/share/temp/model_repos/internlm-chat-7b /root/model/ailab
```

### 2.3 代码准备

```bash
mkdir /root/code && cd /root/code
git clone https://gitee.com/internlm/InternLM.git
cd InternLM
# 与教程 commit 版本保持一致
git checkout 3028f07cb79e5b1d7342f4ad8d11efad3fd13d17 -b demo
```

    将/root/code/InternLM/web_demo.py中 29 行和 33 行的模型路径更换为本地的 /root/model/ailab/internlm-chat-7b。

### 2.4 终端运行

    接着用上面的环境，在/root/code/InternLM目录下新建cli_demo.py文件，内容如下：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/model/ailab/internlm-chat-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("User  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break
    response, history = model.chat(tokenizer, input_text, history=messages)
    messages.append((input_text, response))
    print(f"robot >>> {response}")
```

    在终端运行下面命令即可体验 InternLM-Chat-7B模型的对话能力。

```bash
python /root/code/InternLM/cli_demo.py
```

    体验结束后用Ctrl+C终止命令运行以释放显卡资源。 

### 2.5 网页运行

    在终端中运行下面命令启动web服务：

```bash
cd /root/code/InternLM
streamlit run web_demo.py --server.address 127.0.0.1 --server.port 6006
```

    在 [InternStudio](https://studio.intern-ai.org.cn/) 中配置好SSH Key，在本地终端用下面命令配置SSH端口转发：

```bash
# 把本地的6006端口转发到远程开发机web_demo的6006端口
# 33090要根据开发机的SSH端口进行更改
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 33090
```

    在浏览器中打开 http://127.0.0.1:6006 页面，此时模型开始加载。在加载完模型之后，就可以与InternLM-Chat-7B进行对话了。体验结束后用Ctrl+C终止命令运行以释放显卡资源。

## 3. Lagent智能体工具调用

### 3.1 环境准备和模型下载

    接着用上面的环境和已经下载的internlm-chat-7b模型。

### 3.2 Lagent 安装

```bash
cd /root/code
# 拉取lagent源码
git clone https://gitee.com/internlm/lagent.git
cd lagent
# 尽量保证和教程commit版本一致
git checkout 511b03889010c4811b1701abb153e02b8e94fb5e -b demo
# 安装依赖
pip install -e .
```

### 3.3 修改代码

    将/root/code/lagent/examples/react_web_demo.py内容替换为以下代码：

```python
import copy
import os

import streamlit as st
from streamlit.logger import get_logger

from lagent.actions import ActionExecutor, GoogleSearch, PythonInterpreter
from lagent.agents.react import ReAct
from lagent.llms import GPTAPI
from lagent.llms.huggingface import HFTransformerCasualLM


class SessionState:

    def init_state(self):
        """Initialize session state variables."""
        st.session_state['assistant'] = []
        st.session_state['user'] = []

        #action_list = [PythonInterpreter(), GoogleSearch()]
        action_list = [PythonInterpreter()]
        st.session_state['plugin_map'] = {
            action.name: action
            for action in action_list
        }
        st.session_state['model_map'] = {}
        st.session_state['model_selected'] = None
        st.session_state['plugin_actions'] = set()

    def clear_state(self):
        """Clear the existing session state."""
        st.session_state['assistant'] = []
        st.session_state['user'] = []
        st.session_state['model_selected'] = None
        if 'chatbot' in st.session_state:
            st.session_state['chatbot']._session_history = []


class StreamlitUI:

    def __init__(self, session_state: SessionState):
        self.init_streamlit()
        self.session_state = session_state

    def init_streamlit(self):
        """Initialize Streamlit's UI settings."""
        st.set_page_config(
            layout='wide',
            page_title='lagent-web',
            page_icon='./docs/imgs/lagent_icon.png')
        # st.header(':robot_face: :blue[Lagent] Web Demo ', divider='rainbow')
        st.sidebar.title('模型控制')

    def setup_sidebar(self):
        """Setup the sidebar for model and plugin selection."""
        model_name = st.sidebar.selectbox(
            '模型选择：', options=['gpt-3.5-turbo','internlm'])
        if model_name != st.session_state['model_selected']:
            model = self.init_model(model_name)
            self.session_state.clear_state()
            st.session_state['model_selected'] = model_name
            if 'chatbot' in st.session_state:
                del st.session_state['chatbot']
        else:
            model = st.session_state['model_map'][model_name]

        plugin_name = st.sidebar.multiselect(
            '插件选择',
            options=list(st.session_state['plugin_map'].keys()),
            default=[list(st.session_state['plugin_map'].keys())[0]],
        )

        plugin_action = [
            st.session_state['plugin_map'][name] for name in plugin_name
        ]
        if 'chatbot' in st.session_state:
            st.session_state['chatbot']._action_executor = ActionExecutor(
                actions=plugin_action)
        if st.sidebar.button('清空对话', key='clear'):
            self.session_state.clear_state()
        uploaded_file = st.sidebar.file_uploader(
            '上传文件', type=['png', 'jpg', 'jpeg', 'mp4', 'mp3', 'wav'])
        return model_name, model, plugin_action, uploaded_file

    def init_model(self, option):
        """Initialize the model based on the selected option."""
        if option not in st.session_state['model_map']:
            if option.startswith('gpt'):
                st.session_state['model_map'][option] = GPTAPI(
                    model_type=option)
            else:
                st.session_state['model_map'][option] = HFTransformerCasualLM(
                    '/root/model/ailab/internlm-chat-7b')
        return st.session_state['model_map'][option]

    def initialize_chatbot(self, model, plugin_action):
        """Initialize the chatbot with the given model and plugin actions."""
        return ReAct(
            llm=model, action_executor=ActionExecutor(actions=plugin_action))

    def render_user(self, prompt: str):
        with st.chat_message('user'):
            st.markdown(prompt)

    def render_assistant(self, agent_return):
        with st.chat_message('assistant'):
            for action in agent_return.actions:
                if (action):
                    self.render_action(action)
            st.markdown(agent_return.response)

    def render_action(self, action):
        with st.expander(action.type, expanded=True):
            st.markdown(
                "<p style='text-align: left;display:flex;'> <span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'>插    件</span><span style='width:14px;text-align:left;display:block;'>:</span><span style='flex:1;'>"  # noqa E501
                + action.type + '</span></p>',
                unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align: left;display:flex;'> <span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'>思考步骤</span><span style='width:14px;text-align:left;display:block;'>:</span><span style='flex:1;'>"  # noqa E501
                + action.thought + '</span></p>',
                unsafe_allow_html=True)
            if (isinstance(action.args, dict) and 'text' in action.args):
                st.markdown(
                    "<p style='text-align: left;display:flex;'><span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'> 执行内容</span><span style='width:14px;text-align:left;display:block;'>:</span></p>",  # noqa E501
                    unsafe_allow_html=True)
                st.markdown(action.args['text'])
            self.render_action_results(action)

    def render_action_results(self, action):
        """Render the results of action, including text, images, videos, and
        audios."""
        if (isinstance(action.result, dict)):
            st.markdown(
                "<p style='text-align: left;display:flex;'><span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'> 执行结果</span><span style='width:14px;text-align:left;display:block;'>:</span></p>",  # noqa E501
                unsafe_allow_html=True)
            if 'text' in action.result:
                st.markdown(
                    "<p style='text-align: left;'>" + action.result['text'] +
                    '</p>',
                    unsafe_allow_html=True)
            if 'image' in action.result:
                image_path = action.result['image']
                image_data = open(image_path, 'rb').read()
                st.image(image_data, caption='Generated Image')
            if 'video' in action.result:
                video_data = action.result['video']
                video_data = open(video_data, 'rb').read()
                st.video(video_data)
            if 'audio' in action.result:
                audio_data = action.result['audio']
                audio_data = open(audio_data, 'rb').read()
                st.audio(audio_data)


def main():
    logger = get_logger(__name__)
    # Initialize Streamlit UI and setup sidebar
    if 'ui' not in st.session_state:
        session_state = SessionState()
        session_state.init_state()
        st.session_state['ui'] = StreamlitUI(session_state)

    else:
        st.set_page_config(
            layout='wide',
            page_title='lagent-web',
            page_icon='./docs/imgs/lagent_icon.png')
        # st.header(':robot_face: :blue[Lagent] Web Demo ', divider='rainbow')
    model_name, model, plugin_action, uploaded_file = st.session_state[
        'ui'].setup_sidebar()

    # Initialize chatbot if it is not already initialized
    # or if the model has changed
    if 'chatbot' not in st.session_state or model != st.session_state[
            'chatbot']._llm:
        st.session_state['chatbot'] = st.session_state[
            'ui'].initialize_chatbot(model, plugin_action)

    for prompt, agent_return in zip(st.session_state['user'],
                                    st.session_state['assistant']):
        st.session_state['ui'].render_user(prompt)
        st.session_state['ui'].render_assistant(agent_return)
    # User input form at the bottom (this part will be at the bottom)
    # with st.form(key='my_form', clear_on_submit=True):

    if user_input := st.chat_input(''):
        st.session_state['ui'].render_user(user_input)
        st.session_state['user'].append(user_input)
        # Add file uploader to sidebar
        if uploaded_file:
            file_bytes = uploaded_file.read()
            file_type = uploaded_file.type
            if 'image' in file_type:
                st.image(file_bytes, caption='Uploaded Image')
            elif 'video' in file_type:
                st.video(file_bytes, caption='Uploaded Video')
            elif 'audio' in file_type:
                st.audio(file_bytes, caption='Uploaded Audio')
            # Save the file to a temporary location and get the path
            file_path = os.path.join(root_dir, uploaded_file.name)
            with open(file_path, 'wb') as tmpfile:
                tmpfile.write(file_bytes)
            st.write(f'File saved at: {file_path}')
            user_input = '我上传了一个图像，路径为: {file_path}. {user_input}'.format(
                file_path=file_path, user_input=user_input)
        agent_return = st.session_state['chatbot'].chat(user_input)
        st.session_state['assistant'].append(copy.deepcopy(agent_return))
        logger.info(agent_return.inner_steps)
        st.session_state['ui'].render_assistant(agent_return)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.join(root_dir, 'tmp_dir')
    os.makedirs(root_dir, exist_ok=True)
    main()
```

### 3.4 运行演示

    运行下面命令启动网页服务：

```bash
streamlit run /root/code/lagent/examples/react_web_demo.py --server.address 127.0.0.1 --server.port 6006
```

    继续运行前面提到的SSH端口转发，在本地浏览器打开 http://127.0.0.1:6006 。在Web页面的模型选择里选internlm，等待模型加载完成后就可以输入一些数学问题来测试了。InternLM-Chat-7B模型会理解题意并生成解题的Python代码，Lagent调度送入Python代码解释器求出该问题的解。

    体验结束后用Ctrl+C终止命令运行以释放显卡资源。

## 4. 浦语·灵笔图文理解创作

### 4.1 环境准备

    在 [InternStudio](https://studio.intern-ai.org.cn/) 开发机的升降配置里选择A100(1/4)*2的配置，待启动后在本地终端通过SSH连接到开发机。执行以下命令：

```bash
# 进入conda环境
bash
# 从本地克隆一个已有的pytorch 2.0.1 的环境
/root/share/install_conda_env_internlm_base.sh xcomposer-demo
# 激活环境
conda activate xcomposer-demo
# 安装依赖
pip install transformers==4.33.1
pip install timm==0.4.12
pip install sentencepiece==0.1.99
pip install gradio==3.44.4
pip install markdown2==2.4.10
pip install xlsxwriter==3.1.2
pip install einops
pip install accelerate
```

### 4.2 模型下载

    [InternStudio](https://studio.intern-ai.org.cn/)平台的share目录下已经为准备了全系列的InternLM模型，直接复制即可。

```bash
cp -r /root/share/temp/model_repos/internlm-xcomposer-7b /root/model/ailab
```

### 4.3 代码准备

```bash
cd /root/code
git clone https://gitee.com/internlm/InternLM-XComposer.git
cd InternLM-XComposer
# 最好保证和教程的 commit 版本一致
git checkout 3e8c79051a1356b9c388a6447867355c0634932d -b demo
```

### 4.4 运行演示

```bash
python examples/web_demo.py  \
    --folder /root/model/ailab/internlm-xcomposer-7b \
    --num_gpus 1 \
    --port 6006
```

    在本地终端执行上面提到的SSH端口转发后，在浏览器打开 http://127.0.0.1:6006 ，等待模型加载后即可体验图文理解创作。

## 5. 其他

### 5.1 PyPI和Anaconda换源

    PyPI镜像使用帮助可参考 https://help.mirrors.cernet.edu.cn/pypi/

    Anaconda镜像使用帮助可参考[Anaconda 软件仓库镜像使用帮助 - MirrorZ Help](https://help.mirrors.cernet.edu.cn/anaconda/)

### 5.2 模型下载

    前面的演示里开发机里已经提供了各种模型，但全新的环境里则需要从各模型提供网站下载。下面列举从三个模型网站下载的方法：

#### 5.3.1 Hugging Face

    使用Hugging Face官方提供的huggingface-cli命令行工具。安装依赖：

```bash
pip install -U huggingface_hub
```

    下载整个模型：

```bash
huggingface-cli download --resume-download \
                         --local-dir your_local_path \
                         internlm/internlm-chat-7b
```

    下载部分文件：

```bash
huggingface-cli download --resume-download \
                         --local-dir your_local_path \
                         internlm/internlm-chat-20b \
                         config.json README.md
```

    也可用Python API来下载，详见[Hugging Face Guides/Download](https://huggingface.co/docs/huggingface_hub/guides/download)

#### 5.3.2 ModelScope

    安装依赖：

```bash
pip install -U modelscope
```

    使用modelscope中的snapshot_download函数下载模型，第一个参数为模型名称，参数 cache_dir为模型的下载路径。

```python
from modelscope import snapshot_download
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b',
                               cache_dir='your_local_path',
                               revision='master')
```

    详见[ModelScope模型下载](https://www.modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E4%B8%8B%E8%BD%BD)

#### 5.3.3 OpenXLab

    安装依赖：

```bash
pip install -U openxlab
```

    安装完成后使用download函数导入模型中心的模型：

```python
from openxlab.model import download
download(model_repo='OpenLMLab/InternLM-7b', 
         model_name='InternLM-7b', output='your_local_path')
```

    详见[下载模型 | OpenXLab浦源 - 文档中心](https://openxlab.org.cn/docs/models/%E4%B8%8B%E8%BD%BD%E6%A8%A1%E5%9E%8B.html)


