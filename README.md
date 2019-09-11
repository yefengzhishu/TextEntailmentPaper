# TextEntailmentPaper
TextEntailment论文分享
## Catalog
* Decomposiable Attention
* ESIM
* SAN
* MTP
* Conclusion

## Decomposiable Attention
### Input and Outputs
* 训练数据：![DA_label](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA输入输出.png)
* 输入：前提Premise![DA_label](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_premise.png)、假说hypothesis![DA_label](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_hypothesis.png)
* 输出：a和b之间的关系标签![DA_label](https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_Label.png),C为输出类别的个数,是个C维的0,1向量

### Model Overview
<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_model_overview.png" height="40%" width="40%" ></div>

### Step
* Attend

	计算a和b的attention score
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_attend_attention_score.png" height="40%" width="40%" ></div>
	其中*F*为FFN
	
	进而归一化加权得到句子表示
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_attend_attention_weights.png" height="30%" width="30%" ></div>
	
* Compare
	
	对加权后的一个句子与另一个原始句子进行比较
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_compare.png" height="40%" width="40%" ></div>
	
* Aggregate

	将Compare得到的向量结合
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_aggregate_add.png" height="40%" width="40%" ></div>
	
	通过分类器预测结果标签
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_aggregate_clasification.png" height="20%" width="20%" ></div>

* Intra-Sentence Attention(optional)
	
	可以在每个句子中使用句子内的attention方式来加强输入词语的语义信息
	
	计算句子内的词之间的attention并得到每个词的表示
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_intra_attention.png" height="30%" width="30%" ></div>
	
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_intra_attention.png" height="30%" width="30%" ></div>
	*d_i-j*表示当前字i与其他字j之间的距离偏差
	
	最终每一个时刻的输入就变为原始输入跟self-attention后的值的拼接所得到的向量
	
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_intra_final_input.png" height="40%" width="40%" ></div>

### Loss Function
多分类的交叉熵损失函数

<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_loss_function.png" height="40%" width="40%" ></div>

### Experiments Result
<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/DA_experiments.png" height="40%" width="40%" ></div>


## ESIM
### Model Overview
<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/ESIM_model_overview.png" height="40%" width="40%" ></div>

### Step
* Input Encoding

	使用复用的BiLSTM单元分别对Premise和Hypothesis进行编码，得到句子的表示
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/ESIM_input_encoding.png" height="40%" width="40%" ></div>
	
* Local Inference Modeling

	计算相似度
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/ESIM_simlilarity.png" height="10%" width="10%" ></div>
	
	同DA类似，归一化加权得到句子表示
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/ESIM_sentence_representation.png" height="40%" width="40%" ></div>
	
	对加权后的一个句子与另一个原始句子进行点乘和求差，希望得到更好的两句话之间的关系表示
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/ESIM_local_inference_enhance.png" height="20%" width="20%" ></div>
	
	
* Inference Composition

	再次通过BiLSTM网络提取上下文信息得到*v_a*和*v_b*，并通过max_pooling和avg_pooling层并concatenate起来
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/ESIM_inference_composition.png" height="40%" width="40%" ></div>
	
	最终将*v*通过一层MLP和Softmax得到最终分类结果

### Loss Function
多分类的交叉熵损失函数

### Experiments Result
<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/ESIM_experiments.png" height="60%" width="60%" ></div>


## SAN
### Model Overview
<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_overview.png" height="70%" width="70%" ></div>

### Step
* Lexicon Encoding Layer

	将Word Embeddings和Character Embeddings连接起来，通过position-wise FFN得到Lexicon Embedding
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_lexicon_embedding.png" height="40%" width="40%" ></div>

* Contextual Encoding Layer

	通过双层BiLSTM得到的hidden state拼接起来得到Precise和Hypothesis的表示
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_contextual_embedding.png" height="40%" width="40%" ></div>

* Memory Layer

	将p和h的表示通过一层NN后计算attention
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_attention.png" height="40%" width="40%" ></div>
	
	拼接p和h的表示，并再通过一个BiLSTM得到p和h的最终表示
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_memory.png" height="20%" width="20%" ></div>
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_final_representation.png" height="30%" width="30%" ></div>

* Answer module

	经过T步GRU计算，得到最终输出
	
	初始状态*s0*
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_answer_s0.png" height="20%" width="20%" ></div>其中<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_answer_alfa.png" height="25%" width="25%" ></div>
	
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_answer_st.png" height="40%" width="40%" ></div>
	其中，
	<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_answer_xt.png" height="40%" width="40%" ></div>
	
	通过softmax得到每个时间t的各分类概率<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_answer_pt.png" height="40%" width="40%" ></div>
	
	求平均得到最终分类概率，<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_answer_pr.png" height="40%" width="40%" ></div>
	
### Experiments Result

<div align="left"><img src="https://github.com/yefengzhishu/TextEntailmentPaper/blob/master/pic/SAN_experiments.png" height="40%" width="40%" ></div>

## Conclusion
### DA
* 在NLI任务上分解成子问题，并行解决，相比其他模型减少了更多参数，并达到很好的效果
* NLP工作的新思路，不需要句子结构深入建模，通过对齐文本也能达到很好的实验结果
* 更多的是强调两句话的词之间的对应关系,这篇文章提到的模型并没有使用到词在句子中的时序关系，更多的是强调两句话的词之间的对应关系（alignment）
* 本文将NLI任务当做是关键问题，并直接解决这个问题，因此比单独给句子编码有巨大的优势；Bowman等其他方法则更关注问题的泛化，也是针对此来构建模型，适用的场景会比本文模型更广泛
* 没有考虑上下文信息
* 将问题简化为单词间对齐问题
### ESIM
* 与DA类似，通过inter-sentence attention使两句话产生交互
* 加入点积和差值的方式，利于发现p和h的关系
### SAN
* 不止根据输入一次进行判断，**该模型维护一个状态并迭代地改进其预测**，可以对更复杂的推理任务进行建模
* 模型较复杂，多步预测最终才取结果，参数量大