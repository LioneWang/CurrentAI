# CurrentAI
25CurrentAI实习
## 文件结构
### ipynb文件
3个ipynb在Jupyternotebook中运行，借助免费TPU算力，得到的模型上传至huggingface
### py文件
包含one的表示单个prompt的推理
不包含one的表示用整个dataset的prompt进行推理
### Dataset
在huggingface个人账户上，经过reddit的数据处理得到分数最高的5k个数据
