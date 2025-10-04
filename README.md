This repository contains Python code that replicates the experiments from the paper "Metalinguistic Prompting: A Case Study of LLMs' Competence on the English Article System" by Hu & Levy (2023). The goal is to explore the distinction between a language model's underlying linguistic knowledge (competence) and its ability to articulate that knowledge when prompted (performance).

The experiments consistently show that directly measuring a model's raw probability distributions is a more reliable way to assess its core knowledge than asking it natural language questions about language itself.

KEY CONCEPTS
------------

The experiments compare two primary methods for evaluating a model's linguistic knowledge:

1.  Direct Measurement: This approach accesses the model's internal state by looking at the log-probabilities it assigns to words or sentences. A higher probability for a correct/plausible option is taken as evidence of the model's inherent knowledge (competence).

2.  Metalinguistic Prompting: This approach queries the model using natural language questions about the linguistic stimuli. For example, instead of just checking the probability of the next word, we ask, "What word is most likely to come next?". The model's ability to answer this question correctly is a measure of its performance.

EXPERIMENTS
-----------

The code replicates three core experiments from the paper using the `deepseek-ai/deepseek-coder-1.3b-base` and `deepseek-ai/deepseek-coder-6.7b-base` models.

---

### Experiment 1: Word Prediction

* Notebook: `exp1.ipynb`
* Objective: To compare the model's confidence (log-probability) in predicting a correct word directly versus when prompted metalinguistically.
* Finding: The model is generally more "confident" (assigns a higher log-probability) when predicting the next word directly. Wrapping the task in a metalinguistic question introduces a performance factor that can obscure its core predictive competence.

[Experiment 1 Results Chart]

---

### Experiment 2: Semantic Plausibility

* Notebook: `exp2.ipynb`
* Objective: To test the model's ability to distinguish between a plausible and an implausible sentence continuation.
* Finding: Direct measurement is the most accurate method. It correctly identifies the more plausible word ~72% of the time. Metalinguistic prompts are less accurate (54-66%), demonstrating that asking the model to perform a comparative judgment is less reliable than checking its raw probabilities.

[Experiment 2 Results Chart]

---

### Experiment 3: Syntactic Judgment (BLiMP)

This experiment tests the model's knowledge of grammatical rules by comparing grammatical and ungrammatical sentences.

#### Experiment 3a: Sentence Judgment (in isolation)

* Notebooks: `exp3a.ipynb`, `exp3a_larger.ipynb`
* Methodology: The model evaluates each sentence individually.
    * Direct: Compares the full log-probability of the grammatical vs. the ungrammatical sentence.
    * Metalinguistic: Asks "Is the following sentence a good sentence of English?" for each sentence and checks the probability of the model answering "Yes" or "No".
* Finding: The direct method is highly accurate (~82-83%). However, the metalinguistic method performs poorly, with accuracy hovering near chance (~50%). The model often fails to report its underlying grammatical knowledge correctly when judging sentences in isolation.

#### Experiment 3b: Sentence Comparison (in pairs)

* Notebooks: `exp3b.ipynb`, `exp3b_larger.ipynb`
* Methodology: The model is shown both the grammatical and ungrammatical sentences at the same time and is asked to choose which one is better.
* Finding: The direct method remains accurate (~74-82%). Crucially, the accuracy of metalinguistic prompting is lower than the direct method but improves significantly compared to Experiment 3a when the task is framed as a direct comparison. This highlights that the format of the prompt is critical to the model's ability to demonstrate its knowledge.

[Experiment 3b Results Chart]

---

SETUP AND USAGE
---------------

### 1. Prerequisites

First, install the required Python libraries. The larger models use 4-bit quantization to reduce memory usage, which requires `bitsandbytes` and `accelerate`.

```bash
pip install torch transformers pandas scipy tqdm matplotlib seaborn accelerate bitsandbytes