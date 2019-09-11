# TextEntailmentPaper
TextEntailment论文分享
## Catalog
* Decomposiable Attention
* ESIM
* SAN
* MTP

## Decomposiable Attention
### 模型输入输出
* 训练数据：![DA_label](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA输入输出.png)
* 输入：前提Premise![DA_label](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_premise.png)、假说hypothesis![DA_label](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_hypothesis.png)
* 输出：a和b之间的关系标签![DA_label](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_Label.png),C为输出类别的个数,是个C维的0,1向量

### Model Overview
![DA_model_overview](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_model_overview.png)

### Step
* Attend

	计算a和b的attention score![DA_attend_attention_score](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_attend_attention_score.png)
	其中*F*为FFN
	
	进而归一化加权得到表示![DA_attend_attention_weights](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_attend_attention_weights.png)
	
* Compare
	
	对加权后的一个句子与另一个原始句子进行比较
	![DA_attend_attention_weights](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_attend_attention_weights.png)
	
* Aggregate

	将Compare得到的向量结合
	![DA_aggregate_add](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_aggregate_add.png)
	
	通过分类器预测结果标签
	![DA_aggregate_clasification](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_aggregate_clasification.png)

* Intra-Sentence Attention(optional)

### Loss Function
多分类的交叉熵损失函数

![DA_loss_function](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_loss_function.png)



## SAN
### Abstract
不止根据输入一次进行判断，**该模型维护一个状态并迭代地改进其预测**
###Model
模型整体流程图如下：
![SAN模型整体流程图](http://d.pr/i/9HM6+)