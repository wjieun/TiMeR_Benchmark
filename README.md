# TiMeR Benchmark

**TiMeR Benchmark** is a benchmark designed to evaluate the performance of language models on **Temporal Deixis Resolution**.\
Follow the instructions below to get started.

## 🛠️ Setup

**Create and activate a new conda environment**

```bash
conda create -n timer python=3.12
conda activate timer
```

**Install the required dependencies**

```bash
pip install -r requirements.txt
```

**Login to Hugging Face**

```bash
huggingface-cli login
```


## ⚙️ Configuration

Open `benchmark.py` and modify the model settings:

```python
batch_size = 16
basemodel_name = "meta-llama/Llama-3.1-8B"
model_name = ""  # Replace with your model name
```

- `batch_size`: Number of inputs processed at once.
- `basemodel_name`: Name of the baseline model used for comparison.
- `model_name`: Your model name from Hugging Face or local environment. Replace the empty string with the desired model.


## 🚀 Run

Once everything is ready, start the benchmarking process:

```bash
python benchmark.py
```