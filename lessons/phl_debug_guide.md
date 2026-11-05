# MetaQA-2hop PHL调试数据

这是一份已运行、已校验的教师参考数据构造，用来调试后续ESIM训练。它不是作者发布的PHL，也不代表论文复现或模型训练已完成。学生仍需独立实现、解释和验证正式提交内容。

## 已生成的数据

输出位置：`data/processed/metaqa_2hop_debug/`，整个目录由现有`.gitignore`中的`data/`规则忽略。

| 划分 | 原始问题数 | PHL条数 | 正例label=0 | 负例label=1 |
|---|---:|---:|---:|---:|
| Train | 500 | 16,561 | 3,301 | 13,260 |
| Dev | 100 | 2,874 | 445 | 2,429 |

原始问题按官方文件前N行选取，保留`train:原始行号`/`dev:原始行号`。没有读取Test问答文件。该调试前缀覆盖21类题型，不等于随机代表完整数据集。

600题的全部3,746个标准答案都在候选中，且均保留相应证据。这里的100%是候选召回，不是模型准确率。Train/Dev原问题文本没有完全相同的重叠；并不声称实体完全隔离。

## 文件与字段

- `train.jsonl` / `dev.jsonl`：每行一个问题-候选样本。模型读`premise`与`hypothesis`，监督目标读`label`。
- `*.questions.jsonl`：原始问题、答案集合、完整候选及保留候选、缺失答案、原始行号；供题级聚合与QA评价。
- `preview.json`：可直接查看正负样本、多路径样本。
- `vocab.json`：只使用Train的P/H文本建立的11,369项词表，`<pad>=0`、`<unk>=1`，保留大小写。
- `batch_example.json`：实际8样本的IDs、mask和label；不是模型预测。
- `validation.json`：逐行数据校验、候选覆盖、文本重叠、OOV与batch形状。
- `manifest.json`：配置、代码/数据SHA256、基础Git提交、运行时间、数据统计与实现边界。

每条PHL保留`question_id`、`split`、`candidate`、`premise_sentences`、`premise_facts`、`hypothesis_rule`及完整`paths`。路径中每条边保存source/target、relation、inverse、原始triple与kb.txt行号。Gold答案不放进模型输入文本。

## 构造协议：哪些是我们的选择

1. 使用MetaQA方括号给出的主题实体。
2. 候选来自恰好两跳的简单路径终点，第一跳限于电影-导演/编剧/演员关系，允许逆向遍历；不读取答案和qtype文件。这个全局schema适合当前21类两跳题，避免从年份/类型/标签枢纽生成大量无关候选，是明确适配，不冒称作者完全公开了同一算法。
3. 候选确定以后，从完整KG的所有关系中收集到该候选的1/2-hop简单路径。证据搜索不只限于候选生成所用的第一跳关系。
4. 当前不抽负例、不限每个候选的路径数；Train与Dev均保留全部检索候选。可选负采样仅对Train生效，并保留全部检索到的正候选；Dev始终保留全部候选。
5. 每条不同原始三元组按关系模板写成一句英语，同一Premise内重复事实去重；完整路径集合单独保留。存储方向是主题实体→候选，原始事实方向不变；后续HRPE需按复现协议明确是否反转路径顺序。
6. 从问题文本推断21类vanilla语义形式，生成存在性陈述。例如“Sergei Kozlov wrote a movie directed by Yuriy Norshteyn.”。官方qtype仅在独立审计中对照转换是否正确，生成器不读取它，也不把关系答案链输入模型。
7. Label按候选是否在该题原始答案集合中生成：0支持、1负候选。类1不必是严格逻辑矛盾，可能仅是与问题无关的真实事实。
8. 问题所属官方split不变，同题全部候选留在同一份。词表仅用Train，Dev未知词映射`<unk>`，当前Dev词级OOV约7.55%。

## 如何读取成batch

```python
import json
from src.phl import make_batch

with open("data/processed/metaqa_2hop_debug/train.jsonl", encoding="utf-8") as f:
    rows = [json.loads(next(f)) for _ in range(8)]
with open("data/processed/metaqa_2hop_debug/vocab.json", encoding="utf-8") as f:
    vocabulary = json.load(f)["token_to_id"]
batch = make_batch(rows, vocabulary)
```

返回`premise_ids`、`premise_mask`、`hypothesis_ids`、`hypothesis_mask`、`labels`，可直接转为PyTorch张量。

当前示例形状是Premise `[8,43]`、Hypothesis `[8,11]`、标签`[8]`。JSONL原文不裁剪；batch默认上限512/128，Train有2条Premise超过512词，Dev没有。`make_batch`显式返回裁剪计数；需要保留这两条全文时把`max_premise`调大。原文与路径始终完整保留。

当前词表与IDs是普通文本NLI接口。论文所用TransE/GloVe、KG实体/关系映射与模型尚未完成，不能把随机词向量训练称为已复现论文原始表示。

## 重建和校验

```powershell
python -s -X utf8 scripts/build_phl.py
python -s -X utf8 scripts/build_phl.py --overwrite
python -s -X utf8 -m unittest discover -s tests -v
```

配置在`configs/phl_debug.json`；可修改问题数量、输出目录、负例比例、每候选路径上限、批处理长度与关系文本模板。生成器独立回读全部19,435条记录，检查标签、原始事实行号、方向、路径连通性和节点不重复、假设与证据文本、路径保留计数、问题级split。未知问句模板会明确报错，不静默生成错误假设。
