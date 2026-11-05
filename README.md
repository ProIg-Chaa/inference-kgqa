# Inference-Based KGQA Reproduction

Reproduction project for COMP9444 Project 003: Knowledge Graph Question Answering (KGQA).

We follow the general approach of An Inference Approach to Question Answering over Knowledge Graphs (Gupta et al., 2021): represent KGQA as a natural language inference (NLI) task, using paths in a knowledge graph as evidence for candidate answers.

**Current branch:** `lesson02-teacher-rehearsal`, a teacher reference for data loading, relation-preserving graphs and bounded paths. Read the [teacher rehearsal guide](lessons/lesson02_teacher_guide.md), or run `python -s -X utf8 scripts/rehearse_lesson02.py`. This is Lesson 02 teaching code; no neural model training or QA-model evaluation has been completed.

[Open the teacher rehearsal in Google Colab](https://colab.research.google.com/github/ProIg-Chaa/inference-kgqa/blob/lesson02-teacher-rehearsal/notebooks/lesson02_teacher_rehearsal.ipynb)

## Lesson 02 branch roles

This teacher branch contains reference modules, a step-by-step notebook and small tests. `lesson02-student-start` contains scaffold and configuration only; students create their own branches there and implement the lesson modules themselves. Dataset files are stored locally and excluded from Git. Branches do not restrict who can read the reference code.

## Project Goals

Explore the MetaQA dataset and construct a knowledge graph from its triples.

Retrieve multi-hop paths related to a question and candidate answer.

Convert retrieved evidence and questions into premise–hypothesis examples.

Implement, train and evaluate an ESIM-based NLI model as our initial reproduction target.

Analyze errors and results; consider HRPE or another improvement if time permits.

The original paper's published scores are reference results, not results of this project. We will report our own experimental settings and measurements separately.

## Repository Structure

We start small and add files as needed:

~~~text
inference-kgqa/
├── README.md
├── requirements.txt
├── .gitignore
├── src/                    # Dataset, graph, paths, then models
└── notebooks/
    └── experiment.ipynb   # Colab entry point and experiment record
~~~

Later we may add tests/, training scripts and model modules. Avoid committing datasets, checkpoints or local virtual environments.

Local MetaQA text data is stored in `src/dataset/` (ignored by Git), with `kb.txt` and the original `1-hop/`, `2-hop/`, and `3-hop/` directories. The notebook defaults to this location. Downloads include vanilla/NTM text and question types; provenance and file hashes are in the local `download_manifest.json`. Obtain the files from the [official MetaQA source](https://github.com/yuyuz/MetaQA); a Git clone does not include them.

## Development Workflow

GitHub: shared codebase, commits and code review.

VS Code + AI coding assistant: implement and understand small modules; review generated code before accepting it.

Google Colab: run experiments in a common Linux environment (GPU only when needed).

Google Drive: store checkpoints and large intermediate outputs.

The student should be able to explain each important function's inputs, outputs and logic, even when an AI assistant helps write it.

## Getting Started

### Local development (Windows or macOS)

~~~bash
git clone --branch lesson02-teacher-rehearsal https://github.com/ProIg-Chaa/inference-kgqa.git
cd inference-kgqa
python -m venv .venv
~~~

Activate the environment and install requirements:

macOS:

~~~bash
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

Windows (PowerShell):

~~~powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

Stage 1-2 currently uses the Python standard library. Add model dependencies when the corresponding implementation is introduced. If a local notebook kernel is needed, install ipykernel in the selected environment. An existing isolated micromamba environment can be used instead of creating a second environment.

### Google Colab

Open [notebooks/experiment.ipynb](notebooks/experiment.ipynb) in Colab. Run the following setup in a new cell before the notebook's configuration cells:

~~~python
!git clone --branch lesson02-teacher-rehearsal https://github.com/ProIg-Chaa/inference-kgqa.git
%cd /content/inference-kgqa
!pip install -r requirements.txt
~~~

Private repositories need authenticated access. Do not put access tokens directly in shared notebooks. Colab sessions are temporary: save important results outside /content.

## Milestones

Stage

Deliverable

1. Data

Load and inspect MetaQA questions and KG triples

2. Graph

Build adjacency lists and retrieve bounded multi-hop paths

3. NLI

Construct premise–hypothesis training examples and define labels

4. Model

Train and evaluate the initial ESIM-based model

5. Analysis

Compare against reported reference results with matched protocols where possible; inspect failures and limitations

6. Extension (optional)

Implement HRPE or another justified improvement

Current focus: Stage 1–2. No model training is required for the second lesson.

## References

Gupta et al. (2021), [An Inference Approach to Question Answering over Knowledge Graphs](https://arxiv.org/abs/2112.11070).

[MetaQA dataset: official repository](https://github.com/yuyuz/MetaQA).

## Reproducibility and Attribution

Record the dataset split, software versions, random seed, hyperparameters and evaluation metric for every experiment.

Keep train/validation/test data separated; never use gold test answers for candidate generation.

Credit the paper, external implementations and any AI-assisted code according to course requirements.

The final course submission will be assembled into the required notebook and report format.
