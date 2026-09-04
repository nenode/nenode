# N1

N1 is a small, trainable autoregressive language model written in Python and PyTorch. It is a real neural network that learns to predict the next character from context; it is not an intent classifier or a set of hand-written responses.

This is an educational foundation, not a ChatGPT-scale model. Its quality depends on the training data and compute available. Character-level training is intentionally simple, so the next upgrade for serious use would be a subword tokenizer and a much larger dataset.

## Run it

Create an environment and install the project:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Train N1 on the included sample data:

```bash
python -m n1.train --steps 1000
```

Talk to the trained N1 model:

```bash
python -m n1.chat
```

Use your own UTF-8 text by passing a file:

```bash
python -m n1.train --data path/to/your-training.txt --steps 5000
```

## Train on Hugging Face data

The trainer can stream a dataset instead of downloading the full corpus. It caches a configurable prefix locally, then trains the N1 neural language model on random context windows from that cache. This makes large datasets practical, but model quality still depends on compute and architecture size.

For a small public test:

```bash
python -m n1.train \
	--dataset roneneldan/TinyStories \
	--text-column text \
	--max-chars 5000000 \
	--steps 5000
```

For a much larger corpus, use FineWeb-Edu with a bounded local cache:

```bash
python -m n1.train \
	--dataset HuggingFaceFW/fineweb-edu \
	--dataset-config sample-10BT \
	--text-column text \
	--max-chars 500000000 \
	--steps 100000
```

`--max-chars 0` streams until the dataset ends, which may require substantial disk space and RAM. Set a Hugging Face token in the environment only when using a private or gated dataset.

The N1 model learns statistical patterns from the text it sees. It does not automatically browse the web, verify facts, remember conversations between runs, or reason like a person. Those are separate capabilities that can be added around the language model later.

## Project layout

- `n1/model.py`: GPT-style causal Transformer architecture.
- `n1/train.py`: local/Hugging Face data loading, byte encoding, batches, optimization, and N1 checkpoints.
- `n1/chat.py`: checkpoint loading and text generation.
- `data/train.txt`: tiny example corpus for a smoke test.
