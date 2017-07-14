# 基于TensorFlow实现的闲聊机器人

GitHub上实际上有些实现，不过最出名的那个是torch实现的，DeepQA这个项目到是实现的不错，不过是针对英文的。

这个是用TensorFlow实现的sequence to sequence生成模型，代码参考的TensorFlow官方的

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/rnn/translate

这个项目，还有就是DeepQA

https://github.com/Conchylicultor/DeepQA

语料文件是db/dgk_shooter_min.conv，来自于 https://github.com/rustch3n/dgk_lost_conv

参考论文：

Sequence to Sequence Learning with Neural Networks

A Neural Conversational Model

# 依赖

python3 是的这份代码应该不兼容python2吧

numpy 科学运算

sklearn 科学运算

tqdm 进度条

tensorflow 深度学习

大概也就依赖这些，如果只是测试，装一个cpu版本的TensorFlow就行了，也很快。

如果要训练还是要用CUDA，否则肯定超级慢超级慢～～

# 本包的使用说明

本包大体上是用上面提到的官方的translate的demo改的，官方那个是英文到法文的翻译模型

下面的步骤看似很复杂……其实很简单

## 第一步

输入：首先从[这里](https://github.com/rustch3n/dgk_lost_conv)下载一份dgk_shooter_min.conv.zip

输出：然后解压出来dgk_shooter_min.conv文件

## 第二步

在***项目录下***执行decode_conv.py脚本

输入：python3 decode_conv.py

输出：会生成一个sqlite3格式的数据库文件在db/conversation.db

## 第三步

在***项目录下***执行data_utils.py脚本

输入：python3 data_utils.py

输出：会生成一个bucket_dbs目录，里面包含了多个sqlite3格式的数据库，这是将数据按照大小分到不同的buckets里面

例如问题ask的长度小于等于5，并且，输出答案answer长度小于15，就会被放到bucket_5_15_db里面

## 第四步 训练

下面的参数仅仅为了测试，训练次数不多，不会训练出一个好的模型

size: 每层LSTM神经元数量

num_layers: 层数

num_epoch: 训练多少轮（回合）

num_per_epoch: 每轮（回合）训练多少样本

具体参数含义可以参考train.py

输入：

```
./train_model.sh
```

上面这个脚本内容相当于运行：

```
python3 s2s.py \
--size 1024 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 64 \
--num_per_epoch 500000 \
--model_dir ./model/model1
```

```
# --size 1024 在gpu=4GB的机器上面出现内存不足错误，修改为--size 512可以正常训练
# -- 如果是tensorflow1.2.1 cpu only版本，只要内存够大，不需要修改size参数，训练时间较长
python3 s2s.py --size 1024 --num_layers 2 --num_epoch 5 --batch_size 64 --num_per_epoch 500000 --model_dir ./model/model1
```

输出：在 model/model1 目录会输出模型文件，上面的参数大概会生成700MB的模型

如果是GPU训练，尤其用的是<=4GB显存的显卡，很可能OOM(Out Of Memory)，
这个时候就只能调小size，num_layers和batch_size

## 第五步 测试

下面的测试参数应该和上面的训练参数一样，只是最后加了--test true 进入测试模式

输入：

```
./train_model.sh test
```

上面这个脚本命令相当于运行：

```
python3 s2s.py \
--size 1024 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 64 \
--num_per_epoch 500000 \
--model_dir ./model/model1 \
--test true
```

输出：在命令行输入问题，机器人就会回答哦！但是上面这个模型会回答的不是很好……当然可能怎么训练都不是很好，不要太期待～～

# 项目文件

db/chinese.txt 小学生必须掌握的2500个汉字

db/gb2312_level1.txt GB2312编码内的一级字库

db/gb2312_level2.txt GB2312编码内的二级字库

*上面几个汉字文件主要是生成字典用的，我知道一般的办法可能是跑一遍数据库，然后生成词频（字频）之类的，然后自动生成一个词典，不过我就是不想那么做……总觉得那么做感觉不纯洁～～*

db/dictionary.json 字典

# 测试结果

**不同的参数和数据集，结果都可能变化很大，仅供参考**

**下面训练结果是用train_model.sh的参数训练的**

> 你好
你好

> 你好呀
你好

> 你是谁
我是说，我们都是朋友

> 你从哪里来
我不知道

> 你到哪里去
你不是说你不是我的

> 你喜欢我吗？
我喜欢你

> 你吃了吗？
我还没吃饭呢

> 你喜欢喝酒吗？
我不知道

> 你讨厌我吗？
我不想让你失去我的家人

> 你喜欢电影吗？
我喜欢

> 陪我聊天吧
好啊

> 千山万水总是情
你不是说你不是我的错

> 你说话没有逻辑啊
没有

> 一枝红杏出墙来
你知道的

# 其他

很多论文进行 bleu 测试，这个本来是测试翻译模型的，其实对于对话没什么太大意义

不过如果想要，可以加 bleu 参数进行测试，例如

```
./train_model.sh bleu 1000
```

具体可以参考 s2s.py 里面的 test_bleu 函数

最后，这个跟现在的机器人平台，和他们所用的技术其实没啥关系，
如果对于机器人(平台)感兴趣，可以看看[这里](https://github.com/qhduan/ConversationalRobotDesign/blob/master/%E5%90%84%E7%A7%8D%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%B9%B3%E5%8F%B0%E8%B0%83%E7%A0%94.md)

更多问题欢迎与我交流


win10_py36分支，是在win10系统 python3.6.1环境，使用tensorflow_gpu1.2.1版本测试通过。
PyCharm运行日志如下：
```
C:\Python36\python.exe D:/git/DeepLearning/chatbot/Seq2Seq_Chatbot_QA/s2s.py --size 512 --num_layers 2 --num_epoch 5 --batch_size 64 --num_per_epoch 500000 --model_dir ./model/model1
dim:  6865
准备数据
bucket 0 中有数据 506206 条
bucket 1 中有数据 1091400 条
bucket 2 中有数据 726867 条
bucket 3 中有数据 217104 条
共有数据 2541577 条

开启投影：512
Epoch 1:
[====================]  100.0%  500032/500000  loss=2.870  50m28s/50m27s

Epoch 2:
[====================]  100.0%  500032/500000  loss=2.369  49m18s/49m18s

Epoch 3:
[====================]  100.0%  500032/500000  loss=2.221  49m39s/49m39s

Epoch 4:
[====================]  100.0%  500032/500000  loss=2.136  48m45s/48m45s

Epoch 5:
[====================]  100.0%  500032/500000  loss=2.072  45m20s/45m19s


Process finished with exit code 0




C:\Python36\python.exe D:/git/DeepLearning/chatbot/Seq2Seq_Chatbot_QA/s2s.py --size 512 --num_layers 2 --num_epoch 5 --batch_size 64 --num_per_epoch 500000 --model_dir ./model/model1 --test True
dim:  6865

开启投影：512
> 你好
我是说，我们要去哪儿
> 中国很美
我们不知道，我们不是在这里的
> 周末运动吧
是的
> 你是谁
我是个军队
> 你从哪里来
我不知道
> 你到哪里去
我不知道
> 你喜欢我吗？
我喜欢
> 你吃了吗？
我不知道
> 你吃了吗？
我不知道
> 你喜欢喝酒吗？
我喜欢
> 千山万水总是情
我们的生活是一个人
> 你说话没有逻辑啊
你说的是什么
> 一枝红杏出墙来
是的
> 其他
不
> 如果
我们会把它们送回来
> 但是
我们不知道
> 哈哈哈哈
你们在这里干什么，我们走
> 
Process finished with exit code 1


C:\Python36\python.exe D:/git/DeepLearning/chatbot/Seq2Seq_Chatbot_QA/s2s.py --size 512 --num_layers 2 --num_epoch 5 --batch_size 64 --num_per_epoch 500000 --model_dir ./model/model1 --test False --bleu 1000
dim:  6865
准备数据
bucket 0 中有数据 506206 条
bucket 1 中有数据 1091400 条
bucket 2 中有数据 726867 条
bucket 3 中有数据 217104 条
共有数据 2541577 条
开启投影：512
 75%|███████▌  | 750/1000 [01:44<00:36,  6.92it/s]2017-07-14 10:48:23.407970: I c:\tf_jenkins\home\workspace\release-win\m\windows-gpu\py\36\tensorflow\core\common_runtime\gpu\pool_allocator.cc:247] PoolAllocator: After 351571 get requests, put_count=351568 evicted_count=1000 eviction_rate=0.0028444 and unsatisfied allocation rate=0.00313735
2017-07-14 10:48:23.408500: I c:\tf_jenkins\home\workspace\release-win\m\windows-gpu\py\36\tensorflow\core\common_runtime\gpu\pool_allocator.cc:259] Raising pool_size_limit_ from 100 to 110
100%|██████████| 1000/1000 [02:18<00:00,  7.42it/s]
BLUE: 1.63 in 1000 samples
```
