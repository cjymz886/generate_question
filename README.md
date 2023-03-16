
# generate question by keywords

1 关键词生成过程：
=
```
 def extract_keywords(query):
        seg_dict = {}
        for w in jieba.cut(query):
            seg_dict[w] = len(w)
        seg_dict = sorted(seg_dict.items(), key=lambda x:x[1], reverse=True)
        top_dict = jieba.analyse.extract_tags(query, topK=5, withWeight=False, allowPOS=())
        k1 = random.choice([1, 2, 3])
        if top_dict:
            return '，'.join(top_dict[:k1])
        else:
            return '，'.join([item[0] for item in seg_dict[:k1]])
```

2 训练、测试、预测过程
=
python run.py train <br>
python run.py test <br>
python run.py predict <br>



