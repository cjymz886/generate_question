# 问句生成相关任务实践
本项目尝试基于语言模型进行问句生成相关任务，第一版使用了百万级问句在T5模型进行了微调，代码在./t5_tuning目录下；第二版尝试基于gpt-2的进行微调生成，本次更新基于ChatGLM-6b+lora微调一版模型，并新增了微调任务种类。

1 本次进展
=
（1）更新基于ChatGLM-6b+lora微调一版模型 <br>
（2）新生成prompt data，新增prompt学习任务 <br>
（4）整理了query库达到4千万级别，并进行了清洗 <br>



2 微调的任务包括
=
（1）关键词到问句生成 <br>
（2）问句生成关键词 <br>
（3）相似问句生成 <br>
微调的数据集在./chatglm_tuning/data目录下，约10万+；<br>

3 训练情况
=
（1）在prompt_data数据集上，训练了3 epochs；<br>
（2）训练loss，从起初的3.0左右，下降到1.3左右;大约在2epoch后，loss下降已特别缓慢；<br>
（3）训练的参数设置可详细见代码；<br>

4 实践效果
=
#关键词控制生成
```
prompt:帮我生成10条包含[流量,套餐]关键词的问句

answer:
1.移动流量包怎么取消流量套餐
2.手机流量套餐有哪些?价格如何查询呢
3.这个有什么的流量套餐吗?可以换卡了吗
4.有免费的流量套餐吗
5.有没有免费流量套餐推荐一下啊!求大神们解答
6.联通流量超出后需要购买什么流量套餐
7.现在哪个网络套餐送更多流量
8.流量多少合适 选哪种套餐好点
9.电信流量套餐好吗
10.移动流量多吗?买套餐怎么样划算呀!谢谢各位大大回答!!
```

#问句生成关键词
```
prompt:[美国CPI为何暴涨]问句包含哪些关键词

answer:
包含的关键词有: 美国,CPI
```

#相似问句生成
```
prompt:帮我生成9条与[炒股软件哪个好用]相似的问句

answer:
1.请问有什么好的炒股工具可以推荐吗
2.想要在股市上获得成功,应该选择哪个炒股软件呢?请帮忙解答一下!
3.我想知道有哪些好用的炒股应用程序值得大家使用
4.股市新手应该如何选择适合自己的炒股软件呢
5.能否介绍一下哪些炒股软件适合股票投资者
6.我想学习更多有关股市的知识和技能,有没有好的炒股软件可以参考下的呢
7.现在有很多不同的炒股软件可供选择,不知道哪种最实用、最好用
8.对于刚刚入门的人来说,用什么炒股软件比较合适比较好
9.想请教各位高手们,哪些炒股软件是投资必备的好帮手啊!
```

5 结果分析
=
（1）在这版训练效果来看，感觉基于chatglm+lora训练的还是不够充分，对prompt_data拟合度偏低，不确定是否加大轮次还是调整下lora参数；<br>
（2）在做问句生成关键词任务上，有很多重要词还是没识别出来，跟prompt data质量不高有关联；<br>

6 下步计划
=
（1）继续优化chatglm+lora模型；<br>
（2）增加或优化prompt任务；<br>
（3）考虑如何将query库都应用上，或构建一个知识库，将模型跟知识库结合来做query相关的生成任务。<br>


