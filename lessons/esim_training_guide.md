# 使用coetaur0/ESIM训练MetaQA PHL

模型来源：[coetaur0/ESIM](https://github.com/coetaur0/ESIM)，固定提交`65611601ff9f17f76e1f246e8e46b5fc4bee13fc`，Apache-2.0。本地源码在`third_party/ESIM/`，保留完整许可、Git来源和提交；目录被忽略，通过`third_party/esim.lock.json`和拉取脚本重建。

这是第三方的顺序ESIM实现，不是Gupta Inference论文作者的官方代码。直接导入上游`esim.model.ESIM`，未改写模型；保留Embedding、双向编码、双向注意力、拼接/差值/乘积、投影、双向推断组合、平均/最大池化和MLP。

## 本地运行

AusDL课堂目录提供「开始ESIM训练.cmd」和「检查ESIM单步.cmd」，复用现有隔离环境。每次产生带时间戳的输出目录，保留旧实验。只检查已有依赖与固定上游源码，不反复安装包或下载SNLI/GloVe。

独立教师仓库内也可运行以下命令，先激活已有PyTorch环境：

```powershell
python -s -X utf8 scripts/setup_esim.py
python -s -X utf8 scripts/train_esim.py --smoke
python -s -X utf8 scripts/train_esim.py
```

日常固定版本核查用`setup_esim.py --check`，不联网。首次setup用Git拉取固定提交，仅签出模型、脚本、配置与根目录许可等文件，不下载上游大型数据/预训练checkpoint。

输出已存在时训练入口会提示使用新的`--output outputs/新目录`，或显式`--overwrite`。可通过`--epochs N`覆盖轮数；其余设置统一编辑`configs/esim_debug.json`，最终实际配置会写进运行记录。

## 数据和模型适配

- 数据读取我们已验收的Train500题/16,561条PHL和Dev100题/2,874条PHL；词表仅来自Train。
- 上游原训练脚本针对SNLI/MultiNLI及pickle数据；项目训练入口改读PHL JSONL，不改上游模型文件。
- 构造真实长度与动态padding，传入`premises, premises_lengths, hypotheses, hypotheses_lengths`。题号与候选用于汇总，不传入模型；gold只用于训练损失与事后评价。
- 分类头通过上游构造参数`num_classes=2`设置，沿用`0=entailment, 1=not_answer`，不套用三分类SNLI checkpoint。
- 当前是随机、可训练的64维词Embedding，hidden_size=32、dropout=0.2、Adam学习率0.001、batch32、CPU两线程、种子17、默认1轮。全部可改。
- 为类别不平衡，默认只按Train正负数量计算损失权重；可将`class_weights`改为null关闭。
- 训练入口长度上限640/128，本次全部P/H都未裁剪；PHL文件中的完整文本和路径始终保留。

随机词Embedding、小尺寸与现有候选/模板均为调试适配。论文的TransE/GloVe表示尚未接入，不能把本次结果称为完整论文复现或全量MetaQA成绩。当前只验证本地CPU；不声称学生GPU/Colab已验证。

## 训练与评价

每个batch执行前向、加权交叉熵、backward、梯度裁剪与Adam更新；初始Dev和每轮Dev使用eval/no_grad。模型输出支持概率，按配置阈值0.5收集所有支持候选；与每题标准答案集合比较，不能只取最高分。

分类准确率分母是候选PHL条数；QA集合完全匹配分母是原始Dev问题数，即使某题预测为空或没有候选也计入。阈值来自配置，未读取Test；Dev用于模型选择，最终Test仍需另留。按Dev集合完全匹配选择best，相同时比较Dev损失。

`outputs/`中每次保存：

- `config.json`：实际参数。
- `history.json`：初始Dev、各轮Train/Dev损失与指标。
- `best.pt` / `last.pt`：完整上游ESIM参数、优化器、配置、词表哈希及上游版本。
- `dev_candidates.jsonl`：最佳checkpoint对应的候选概率、预测标签与gold标签。
- `dev_questions.jsonl`：每题预测答案、标准答案、漏答、多答和集合匹配。
- `manifest.json`：来源提交/许可/哈希、数据与模型规模、截断、权重及运行边界。
- `source_snapshot/`：启动时保存项目适配代码，防止后续改代码丢失实验来源。

本次验收运行见`outputs/esim_debug_verified/`：完整训练1轮，验证集100题。已有初次完整试跑结果为候选分类90.01%、集合完全匹配48/100；以该目录实际manifest/history为准，不作为固定回归答案。

## 重新加载checkpoint验证

```powershell
python -s -X utf8 scripts/evaluate_esim.py --checkpoint outputs/esim_debug_verified/best.pt
```

脚本核对词表、数据清单、上游源码，重新读取Dev并输出`evaluation_dev/`，检查分类与集合指标能否重现。不会训练、调阈值或读取Test。

测试命令：

```powershell
python -s -X utf8 -m unittest discover -s tests -v
```

在既有隔离PyTorch环境中检查了padding/排序恢复、批量与单样本预测一致、额外padding不改变输出，以及漏答/多答/空预测与原问题分母。无PyTorch的纯数据环境会跳过两个模型适配测试。
