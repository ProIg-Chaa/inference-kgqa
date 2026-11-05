# 第二课教师演练

教师参考分支：`lesson02-teacher-rehearsal`。学生起步分支：`lesson02-student-start`，只保留基础骨架，没有第二课模块答案。

## 本地一遍跑通

在本仓库运行：

```powershell
git switch lesson02-teacher-rehearsal
python -s -X utf8 -m unittest discover -s tests -v
python -s -X utf8 scripts/rehearse_lesson02.py
```

最后一条命令按顺序回放Notebook的Python单元，不需要Jupyter。也可打开 `notebooks/lesson02_teacher_rehearsal.ipynb`，选定已有内核从头执行。演练代码只用标准库，不需要GPU；VS Code交互Notebook本身需要可用的Jupyter内核。本机数据在 `src/dataset/`，需要 `kb.txt` 和 `2-hop/vanilla/qa_train.txt`。

若在其他目录存放数据，设置环境变量 `METAQA_DATA_DIR` 后重启Notebook内核，或修改开头配置单元。Notebook从项目根或notebooks目录均可运行。

## 按课堂顺序演练

| 分钟 | 操作 | 停下来解释什么 |
|---|---|---|
| 0–10 | 明确本课范围，检查Python字典/列表/函数 | 数据与路径在论文框架中的位置 |
| 10–20 | 核对仓库、入口与数据目录 | clone不会带上被忽略的数据 |
| 20–40 | 原始行→QA/三元组，查看load_metaqa.py | TAB、竖线、多答案、问题ID、split |
| 40–65 | 查看build_graph.py，手查邻接表 | 多关系、反向访问与原事实方向 |
| 65–70 | 休息，脱离代码口述 | 一条邻接记录的字段 |
| 70–95 | 小图预判→path_retrieval.py→少量真实问题 | 路径级visited、边数限制、关系保留 |
| 95–110 | 运行小测试，解释报错 | 测试预期不靠Agent给结论 |
| 110–120 | 复盘、提交演练、机动 | 哪些完成、哪些留待下节 |

教师先完整运行一次，再尝试隐藏参考实现，按“描述输入输出→预测→写一小段→运行解释”推进。每次只聚焦一个函数；不要让学生一次粘贴整个参考项目。

从空骨架模拟时可以创建个人练习分支：

```powershell
git switch -c my-lesson02-practice lesson02-student-start
```

此时参考Notebook和模块会随分支切换隐藏；在另一个窗口打开GitHub教师分支作为参考。自己的练习代码保留在个人分支，不合并到学生起步分支。

## Colab课前核验

[打开教师演练Notebook](https://colab.research.google.com/github/ProIg-Chaa/inference-kgqa/blob/lesson02-teacher-rehearsal/notebooks/lesson02_teacher_rehearsal.ipynb)

先在Colab新单元执行：

```python
!git clone --branch lesson02-teacher-rehearsal https://github.com/ProIg-Chaa/inference-kgqa.git
%cd /content/inference-kgqa
```

另行把官方MetaQA文本上传到相同目录结构，或挂载自己的Drive并设置 `METAQA_DATA_DIR`。代码和数据不是同一个同步渠道。先课前验证一遍；本分支的本地运行记录不等于已在Colab运行。课堂无需GPU，可保留本地VS Code作为备用。

## 验收边界

配套逐题讲稿：[Markdown代码解读](lesson02_code_walkthrough.md) / [20页PDF](lesson02_teacher_notes.pdf)。本地AusDL授课工作区可从 `class/11-5/开始第二课.cmd` 打开18主题edtrace讲义；这个启动器依赖工作区既有环境，不包含在独立代码仓库中。

- 读取方括号给定主题实体，不声称训练了实体识别模型。
- 使用已知Train答案作为路径目标是函数验证，未知题候选生成留到下一课。
- 输出带原始三元组和inverse信息；相同节点之间的不同关系不会覆盖。
- 输出上限可能截断路径，不能把返回数量当成完整召回。
- 本课没有NLI样本构造、ESIM/HRPE训练或QA模型评价。
- 两个分支是工作起点的区分，不是权限隔离；有仓库访问权的人可以查看教师分支。
