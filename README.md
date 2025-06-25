# TDBench

This repository contains source code of *Harnessing Temporal Databases for Systematic Evaluation of Factual Time-Sensitive Question-Answering in Large Language Models [arXiv'25]*.

With this repository, you can:
- Continuously evaluate LLMs with latest world knowledge (i.e., temporal aligment task)
- Evaluate how well LLMs can understand temporal constraints in questions (i.e., temporal reasoning task)
- Use your own temporal databases, creating customized Time-Sensitive QA (TSQA) benchmarks


## 📓 Table of Contents

- [Working Tree Overview](#working-tree-Overview)
- [Getting Started](#getting-started)
- [Reproduce Experiments](#reproduce-experiments)
- [Create Your Own Benchmarks](#create-your-own-benchmarks)


## 📁 Working Tree Overview

The `tdbench` repository is structured as follows:

```
tdbench/
├── dataset/                  # Pre-processed tabular data 
├── qa_pairs/                 # Generated TSQA pairs 
├── src/                      # Source code for the project
│   ├── evaluation/             # Evaluation metrics and logic
│   ├── qa_pairs/               # Code related to QA pair generation
│   └── utils/                  # Utility functions
├── run_qa.py                 # Main script to run QA tasks
├── requirements.txt          # Python package dependencies
└── temporal.yaml             # Configuration file for temporal settings or experiments
```


## 🚀 Getting Started
Set up the `tdbench` repository using either `requirements.txt` or the conda-based `temporal.yaml`.

### Option 1: Using `requirements.txt`
Clone the repository and install dependencies:
```
git clone https://github.com/your-username/tdbench.git
cd tdbench
pip install -r requirements.txt
```

### Option 2: Using `temporal.yaml`
If you use conda, just import the environment:
```
git clone https://github.com/your-username/tdbench.git
cd tdbench
conda env create -f temporal.yaml
conda activate temporal
```

## 📊 Reproduce Experiments
As shown in the below figure, TDBench inputs temporal databases and generates TSQA pairs for temporal evaluation of LLMs.
We explain step-by-step procedure for applying TDBench:
1. [how to prepare input datasets](#1-prepare-datasetsrepare-datasets)
2. [how to generate SQL queries from the input datasets](#2-generate-sql-queries)
3. [how to convert SQL queries into natural language TSQA pairs](#3-convert-sql-pairs)
4. [how to perform TSQA tasks to LLMs using the generated TSQA pairs](#4-run-qa-tasks)
5. [how to evaluate LLM responses](#5-evaluate-responses)


### 1. Prepare Datasets
The `dataset/crafted' folder contains the preprocessed tabular datasets used in the paper: 
- culture: *Heritage* dataset
- dyknow: *Country/Athelete/Organization* datasets
- environ: *Carbon Tax* dataset
- legal: *Law* dataset
- medical: *Medical* dataset
- movie: *Netflix* dataset
- olympic: *Olympic* dataset and *Olympic_Country* dataset, a joined dataset for the multi-hop setting.

The metadata including Temporal functional dependencies (TFDs) and time granularity (e.g., year, month) of each dataset are stored in the `utils/utils_dataset.py`.
For the original data sources of the above datasets, please refer to the Section C.2 of the paper.
If you want to add new tabular datasets of your own, please refer to the [Create Your Own Benchmarks](#create-your-own-benchmarks) section.



### 2. Generate SQL queries
By using codes in `src/qa_pairs`, you should first generate SQL queries, which will be translated into natural language TSQA pairs later on. We can terget two different TSQA tasks: temporal alignment and temporal reasoning.


#### 2-1. SQL Queries for Temporal Alignment
Temporal alignment questions ask about current world's knolwedge (e.g., Who is the current president?). To generate such queries, run:
```
cd tdbench/src
python -m qa_pairs.write_sql --task <dataset_name> --qtype now
```

For example, you can run:
```
python -m qa_pairs.write_sql --task culture --qtype now 
```


#### 2-2. SQL Queries for Temporal Reasoning
Compared to the temporal alignment task, temporal reasoning questions incorporates more diverse type of temporal constraints (e.g., before, after, meet, etc.) rather than the temporal constraint of "current". To generate such queries, run:
```
cd tdbench/src
python -m qa_pairs.write_sql --task <dataset_name> --qtype basic
```

The generated SQL queries will be saved in the `tdbench/qa_pairs` folder.


### 3. Convert SQL Pairs into Natural Language QA pairs
After generating SQL queries, you can tranlsate these queries into natural language QA pairs. 

#### 3-1. Set Up the Translator LLM
You should first update `src/utils/utils_llm.py`, adding API keys and download [Ollama](https://github.com/ollama/ollama) to run the LLM APIs.
While we use GPT-4o for our experiments, you can freely change the model type for the translator.

#### 3-2. Tranlsate SQL Queries
Now, run:
```
cd tdbench/src
python -m qa_pairs.convert_sql --task <dataset_name> --qtype <task_type> 
```

The generated QA pairs will be saved in the `qa_pairs` folder, with the name of the translator used.


### 4. Run QA Tasks
Once the QA pairs are ready, we can run QA tasks.

#### 4-1. Set Up LLMs for Evaluation
Add your LLM API keys to `src/utils/utils_dataset.py` to get responses to the LLMs for the evaluation, where we have implemented inference codes for 8 LLMs - *GPT-3.5, GPT4, GPT-4o, Llama3, Mixtral, Gemma2, Qwen2, Granite3* - using [Ollama](https://github.com/ollama/ollama) and [OpenAI API](https://openai.com/index/openai-api/). You may optionally update `run_llm()` to support more LLMs.

#### 4-2. Run QA Task
Run:
```
python run_qa.py --task <dataset_name> --qtype <task_type> --model <LLM_name>
```

The QA result file will be saved in the `tdbench/results` folder.


### 5. Evaluate Responses
Use `tdbench/src/evaluation/analyze_result.py` for the evaluation.
TDBench verify both final answer and time references during the TSQA task.
For example, given LLM responses to temporal alignment questions like "Who is currently serving as the president of U.S.?", TDBench verifies whether both (1) the name of the president and (2) his start date is correctly included in the model response, assessing both the final answer and model explanations.

#### 5-1. Set Up the Time Verifier LLM
While we use rule-based string matching to compute the final answer accuracy, we employ an LLM to compute time accuracy -- please refer to the Section 3.2 of the paper for more details of time accuracy evaluation. Similar to the [translator LLM](#3-1-set-up-the-translator-llm), you can freely change the model type while we use deepseek-R1-14b via [Ollama](https://github.com/ollama/ollama).


#### 5-2. Analyze QA Results
To analyze QA responses, run:
```
cd tdbench/src
python -m evaluation.analyze_result --task <dataset_name> --qtype <task_type> --file <path to the QA log file (.jsonl)>
```

For example, you can run:
```
python -m evaluation.analyze_result --task dyknow --qtype now --file /home/ssoy0701/tdbench/results/dyknow/gpt4o_now_0.jsonl
```

The evaluation results will be saved in the `tdbench/results` folder.


## 🙋 Create Your Own Benchmarks
TDBench offers great flexibility with arbitrary temporal databases. That means, you can:
- Use your own temporal databases, creating customized TSQA benchmarks
- Continuously evaluate LLMs with latest world knowledge by simply refreshing the contents of the input databases

To define new datasets, you should update `utils/utils_dataset.py`, creating new dataset class that inherits the base dataset class: `Mydataset`. Especially, you should sepcify Temporal Functional Dependencies (TFDs) satisfied in your datasets. For example, given the dataset schema Country(country, role, name, start, end), a TFD $country, role \overset{\scriptstyle T}{\rightarrow} name$ denotes a relationship between the dataset attributes -- indicating that if country and role are specified, we can determine the name of the role-holder at any time point (e.g. president of a country). Once you define your dataset class and TFDs, TDBench will leverage them to automatically construct time-sensitive questions -- please refer to the Section 3 of the paper for more details.

In case your temporal databases does not have any TFDs satisfied, refer to the Section B.2 of the paper to extend TDBench by manually defining QA attributes of your dataset.
