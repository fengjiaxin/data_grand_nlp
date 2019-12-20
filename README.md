# 2018“达观杯”文本智能处理挑战赛 代码复现
主要参考如下:

- [Top 10 “万里阳光号”解决方案](https://github.com/moneyDboat/data_grand)
- [“达观杯”文本分类挑战赛Top10经验分享](https://zhuanlan.zhihu.com/p/45391378)

### 环境配置
代码基于Pytorch，版本为0.4.1，Python版本为3.6，torchtext版本为0.2.3 需安装：

- pytorch
- torchtext
- gensim
- pandas
- sklearn
- numpy

### 训练说明
&emsp;&emsp;目前版本的代码只是应用在单机单gpu上，之后探索单机多gpu训练模型。

### 文件说明
```
code/: 模型的代码，包含以下文件
	models/: 深度学习模型
	process_data.py: 数据生成content文件和train val文件划分
	data.py: 数据预处理
	hparams.py: 模型的超参数配置
	train_word2vec: 利用gensim的word2vec模型训练得到词向量
	main.py: 模型训练

notebooks/: 在编程过程中的一些测试代码

data/: 模型的训练文件，包含以下目录：分为all/ 和 mini/ 其中all/存储的是全量数据，mini/是一小部分数据集，方便在电脑调试代码，接下来以data/mini/ 进行说明
	datasets1:原始的训练数据文件夹（必须存在,包含两个）
	content_data:上下文数据文件夹 (文件夹可以自己创建，后续生成两个文件raw_word.txt和raw_article.txt)
	article_data:article文件目录（文件夹可以自己创建，后续生成三个文件train_set.csv,val_set.csv和test_set.csv）
	word_data:word文件目录（文件夹可以自己创建，后续生成三个文件train_set.csv,val_set.csv和test_set.csv）
	result:模型的生成test数据文件目录
	save_models:模型的存储文件夹
	word2vec_data:根据word2vec生成的词向量文件目录
```

### 文本预处理

&emsp;&emsp;首先根据原始文件生成article和word的上下文文件，然后将比赛提供的训练数据按照9:1的比例划分为训练集和验证集。注意需要在process_data.py文件中手动修改文件的位置。

```
python process_data.py
```

### 词／字向量训练

&emsp;&emsp;词／字向量训练使用gensim的word2vec包，自己也手动写了word2vec和glove的模型，但是训练的速度比较慢，效果一般，因此直接采用现有的word2vec包。

主要参考资料如下:

- [用gensim学习word2vec - 刘建平Pinard](https://www.cnblogs.com/pinard/p/7278324.html)
- [基于 Gensim 的 Word2Vec 实践](https://zhuanlan.zhihu.com/p/24961011)

&emsp;&emsp;分别使用训练集和测试集中所有词文本和字文本训练词向量和字向量，向量维度设置为50维。注意需要手动修改在train_word2vec.py文件中的文件夹信息和向量维度信息。运行后可以得到word_50.txt和article_50.txt两个文件。

```
python train_word2vec.py
```

### 全量数据统计 

&emsp;&emsp;word文本平均长度为717，按照覆盖95%样本的标准，取截断长度为2000；article文本平均长度为1177，按同样的标准取截断长度为3200。  

### 数据生成器
&emsp;&emsp;数据生成器起先自己写的，但是占用内存比较大，也可以运行，后来看到有开源的torchtext包，可以直接进行数据生成，其中学习的torchtext教程如下:

- [pytorch学习（十九）：torchtext](http://www.imooc.com/article/32237)
- [torchtext学习总结](https://blog.csdn.net/leo_95/article/details/87708267)
- [torchtext 官方文档](https://torchtext.readthedocs.io/en/latest/index.html)
- [TorchText用法示例及完整代码](https://blog.csdn.net/nlpuser/article/details/88067167)

&emsp;&emsp;从csv文件中提取文本数据，使用torchtext进行文本预处理，并进一步构造batch，这部分代码见data.py的类GrandDataset和方法load_data()。

### 训练模型
主要用到了五个模型

- TextCNN: models/TextCNN.py
- GRU_Attention: models/GRU_Attention.py
- RCNN: models/RCNN.py
- FastText: models/FastText.py
- TextGRU: models/TextGRU  

&emsp;&emsp;分别训练基于word向量和article向量的模型，注意word/article embedding的存放路径， 注意模型配置位于 hparams.py，模型训练代码位于main.py中的main方法，命令示例如下:

```
python main.py --model_name='FastText' --text_type='word' --model_id='word_0'
python main.py --model_name='FastText' --text_type='article' --model_id='article_0'

python main.py --model_name='GRU_Attention.py' --text_type='word' --model_id='word_0'
python main.py --model_name='GRU_Attention.py' --text_type='article' --model_id='article_0'

python main.py --model_name='FastText' --text_type='word' --model_id='word_0'
python main.py --model_name='FastText' --text_type='article' --model_id='article_0'

python main.py --model_name='FastText' --text_type='word' --model_id='word_0'
python main.py --model_name='FastText' --text_type='article' --model_id='article_0'

python main.py --model_name='FastText' --text_type='word' --model_id='word_0'
python main.py --model_name='FastText' --text_type='article' --model_id='article_0'
```

### 训练策略
- 优化器选用torch.optim.Adam，初始学习率设置为1e-3（注意embedding层的学习率另外设置）。
- 先固定embedding层的参数，每个epoch后计算模型在验证集上的f1值，如果开始下降（一般为2-3个epoch之后），将embedding层的学习率设为2e-4。
- 每个epoch后计算验证集上的f1值，如上升则保存当前模型，位于文件夹snapshot/；如果下降则从snapshot中加载当前最好的模型，并降低学习率。
- 如果学习率低于config.py中设置的最低学习率，则终止训练。如果设置的最低学习率为1e-5，一般15个epoch左右后训练终止。


### 后续需要提高的地方
- 1. 尝试更多模型
- 2. 模型融合
- 3. 样本不均衡问题
- 4. fine tune
- 5. 调参数
