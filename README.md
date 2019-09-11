# TextEntailmentPaper
TextEntailment论文分享
## Content
* Decomposiable Attention
* ESIM
* SAN
* MTP

## Decomposiable Attention
### 模型输入输出
* 训练数据：![DA_label](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_Label.png)
* 输入：前提Premise、假说hypothesis
* 输出：标签

###Step
* Attend

	连接

* Compare
* Aggregate
* Intra-Sentence Attention(optional)


## SAN
### Abstract
不止根据输入一次进行判断，**该模型维护一个状态并迭代地改进其预测**
###Model
模型整体流程图如下：
![SAN模型整体流程图](http://d.pr/i/9HM6+)
