# TextEntailmentPaper
TextEntailment论文分享

## Task Goals
解决两个文本之间的推理关系，其中一个文本作为前提（premise），另一个文本作为假设（hypothesis），根据前提P去推断假设H。

如果由P可以推断出H，则P和H为蕴含关系（entailment）；如果P和H矛盾，则为矛盾关系（Contradiction）；如果P和H无关，则为中性关系（Neutral）。

## Catalog
* Decomposable Attention model
* Enhanced Sequential Inference Model（ESIM）
* Stochastic Answer Networks（SAN）
* Multi-Task Deep Neural Networks（MT-DNN）
* Conclusion

## Decomposiable Attention Model
* Paper：https://arxiv.org/abs/1606.01933

### Input and Outputs
* 训练数据：![DA_label](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA输入输出.png)
* 输入：前提Premise![DA_label](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_premise.png)、假说hypothesis![DA_label](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_hypothesis.png)
* 输出：a和b之间的关系标签![DA_label](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_Label.png),C为输出类别的个数,是个C维的0,1向量

### Model Overview
<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_model_overview.png" height="100%" width="100%" ></div>

### Step
* Input Presentation

	原始模型使用每个字的Word Embedding作为输入
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_input_a.png" height="15%" width="15%" ></div>
	
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_input_b.png" height="40%" width="40%" ></div>
	
* Attend

	计算a和b每个字之间的attention score
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_attend_attention_score.png" height="40%" width="40%" ></div>
	其中*F*为FFN
	通过分解的方式可以将对(a,b)通过*lb***lb*次*F'*，分解为通过*la*+*lb*次*F*，并且可以并行执行
	
	进而对attention weight归一化加权得到句子表示
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_attend_attention_weights.png" height="30%" width="30%" ></div>
	*β_i*对应的是*a_i*与b中的每个字对应加权得到的表示
	例：
	今天阳光明媚
	外面天气很晴朗
	
* Compare
	
	对原始句子的字和与另一句每个字对应加权后的表示拼接起来，通过另一个前馈网络G
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_compare.png" height="40%" width="40%" ></div>
	
* Aggregate

	将Compare得到的向量求和
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_aggregate_add.png" height="40%" width="40%" ></div>
	
	通过分类器H预测结果标签
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_aggregate_clasification.png" height="20%" width="20%" ></div>

* Intra-Sentence Attention(optional)
	
	可以使用句子内的attention来增强输入的表示
	
	计算句子内的每个词之间的attention并加入距离偏移*d_i-j*得到每个词的表示
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_intra_attention.png" height="30%" width="30%" ></div>
	
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_intra_attention.png" height="30%" width="30%" ></div>
	
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_intra_attention_sum.png" height="30%" width="30%" ></div>
	
	最终每一个时刻的字表示就变为原始输入跟Intra-Sentence Attention后的值的拼接所得到的向量
	
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_intra_final_input.png" height="30%" width="30%" ></div>

### Loss Function
多分类的交叉熵损失函数

<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_loss_function.png" height="50%" width="50%" ></div>

### Experiments Result
<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_experiments.png" height="80%" width="80%" ></div>

### Conclusion
* 提出一种新思路，不需要对句子结构深入建模，而是通过对齐文本的方式计算相似度
* 在Attend阶段可以通过并行的方式提高速度
* 参数量少，只用了FFN，连RNN都没有用
* 缺乏上下文信息，但是可以通过Intra-Sentence Attention改善


## Enhanced Sequential Inference Model（ESIM）
* Paper：https://arxiv.org/abs/1609.06038

### Model Overview
<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/ESIM_model_overview.png" height="50%" width="50%" ></div>

### Step
* Input Encoding

	a和b分别为句子每个字的Word Embedding
	使用复用的BiLSTM单元分别对a和b进行编码，得到句子中每个字的hidden state
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/ESIM_input_encoding.png" height="35%" width="35%" ></div>
	
* Local Inference Modeling

	计算两个句子每个字表示的相似度
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/ESIM_simlilarity.png" height="15%" width="15%" ></div>
	
	同DA Model类似，归一化加权得到对应的句子表示
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/ESIM_sentence_representation.png" height="40%" width="40%" ></div>
	
	对加权后的对应句子表示与原始句子表示进行点乘和求差，希望得到更好的两句话之间的关系表示
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/ESIM_local_inference_enhance.png" height="25%" width="25%" ></div>
	
	
* Inference Composition

	再次通过BiLSTM网络结合上下文信息得到*v_a*和*v_b*，通过max_pooling和avg_pooling，将结果concatenate起来
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/ESIM_inference_composition.png" height="45%" width="45%" ></div>
	
	最终将*v*通过一层MLP和Softmax得到最终分类结果

### Loss Function
多分类的交叉熵损失函数

### Experiments Result
<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/ESIM_experiments.png" height="60%" width="60%" ></div>

### Conclusion
* 使用BiLSTM结合句子的上下文信息
* 与DA Model类似，通过两个句子每个字之间的相似度使两句话产生交互
* 加入点积和差值的方式，利于发现Premise和Hypothesis之间的关系



## Stochastic Answer Networks（SAN）
* Paper：https://arxiv.org/abs/1804.07888

### Model Overview
<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_overview.png" height="55%" width="55%" ></div>

### Step
* Lexicon Encoding Layer

	将Word Embeddings和Character Embeddings拼接起来，通过position-wise FFN得到Lexicon Embedding
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_lexicon_embedding.png" height="30%" width="30%" ></div>

* Contextual Encoding Layer

	通过双层BiLSTM得到的hidden state拼接起来得到Precise和Hypothesis的表示
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_contextual_embedding.png" height="30%" width="30%" ></div>

* Memory Layer

	将p和h的表示通过一层NN后计算attention
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_attention.png" height="40%" width="40%" ></div>
	
	将p和p与h做attention的表示拼接起来
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_memory.png" height="15%" width="15%" ></div>
	再通过一个BiLSTM得到p和h的最终表示
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_final_representation.png" height="25%" width="25%" ></div>

* Answer module

	* * 经过T步GRU计算，得到最终输出
	
	* * 初始状态*s0*是h的信息的summary
		<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_answer_s0.png" height="20%" width="20%" ></div>其中<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_answer_alfa.png" height="25%" width="25%" ></div>
	* * *s_t*根据之前的状态*s_t-1*和*x_t*进行更新，
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_answer_st.png" height="25%" width="25%" ></div>
	* * 其中，*x_t*由p的信息和历史状态*s_t-1*共同决定
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_answer_xt.png" height="50%" width="50%" ></div>
	
	* * 通过softmax得到每个时间点t的各分类概率<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_answer_pt.png" height="40%" width="40%" ></div>
	
	* * 求平均得到最终分类概率，<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_answer_pr.png" height="25%" width="25%" ></div>
	
### Experiments Result
<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_experiments.png" height="40%" width="40%" ></div>

### Conclusion
* 不同于其他模型根据输入一次进行判断，该模型维护一个状态并迭代地改进其预测，可以对更复杂的推理任务进行建模
* 没有用到Bert、ELMO等词预训练向量


## Multi-Task Deep Neural Networks（MT-DNN）
* Paper：https://arxiv.org/abs/1901.11504

### Model Overview
<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/MTDNN_overview.png" height="70%" width="70%" ></div>

### Modules
#### Shared Layers
* Lexicon Encoder
* Transformer Encoder
#### Task Specific Layers
* Single-Sentence Classification
* Text Similarity
* Pairwise Text Classification
* Relevance Ranking
	
### Experiments Result
<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/MTDNN_experiments.png" height="40%" width="40%" ></div>

### MT-DNN
* 达到目前SNLI的start-of-the-art
* 可以利用相关的多个任务的标注数据来训练.
* 可以避免发生对一个特定任务出现过拟合
* 训练复杂，参数量大


## Conclusion
### SNLI排行榜
<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/conclusion_compare.png" height="100%" width="100%" ></div>

### DA
* 提出一种新思路，不需要对句子结构深入建模，而是通过对齐文本的方式计算相似度
* 在Attend阶段可以通过并行的方式提高速度
* 参数量少，只用了FFN，连RNN都没有用
* 缺乏上下文信息，但是可以通过Intra-Sentence Attention改善
### ESIM
* 使用BiLSTM结合句子的上下文信息
* 与DA Model类似，通过两个句子每个字之间的相似度使两句话产生交互
* 加入点积和差值的方式，利于发现Premise和Hypothesis之间的关系
### SAN
* 不同于其他模型根据输入一次进行判断，该模型维护一个状态并迭代地改进其预测，可以对更复杂的推理任务进行建模
* 没有用到Bert、ELMO等词预训练向量
### MT-DNN
* 达到目前SNLI的start-of-the-art
* 可以利用相关的多个任务的标注数据来训练.
* 可以避免发生对一个特定任务出现过拟合
* 训练复杂，参数量大