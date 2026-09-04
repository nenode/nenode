import argparse
import random
from pathlib import Path

import torch

from .model import GPT, GPTConfig


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    inputs = torch.stack([data[start : start + block_size] for start in starts])
    targets = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return inputs.to(device), targets.to(device)


def load_text(args: argparse.Namespace) -> str:
    if args.dataset:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "Hugging Face datasets is required for --dataset; install the project dependencies first"
            ) from error

        dataset_args = {"path": args.dataset, "split": args.split, "streaming": True}
        if args.dataset_config:
            dataset_args["name"] = args.dataset_config
        stream = load_dataset(**dataset_args)
        text = bytearray()
        for row in stream:
            value = row.get(args.text_column)
            if not isinstance(value, str):
                continue
            text.extend(value.encode("utf-8"))
            text.extend(b"\n")
            if args.max_chars and len(text) >= args.max_chars:
                break
        return bytes(text).decode("latin-1")

    return args.data.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Nenode on local text or a streaming Hugging Face dataset.")
    parser.add_argument("--data", type=Path, default=Path("data/train.txt"))
    parser.add_argument("--dataset", help="Hugging Face dataset ID, for example HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", help="Dataset configuration/name, when the dataset has multiple configs")
    parser.add_argument("--split", default="train", help="Dataset split to stream")
    parser.add_argument("--text-column", default="text", help="Dataset column containing training text")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=50_000_000,
        help="Maximum UTF-8 bytes to cache from the stream (0 means unlimited)",
    )
    parser.add_argument("--out", type=Path, default=Path("checkpoints/nenode.pt"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    text = load_text(args)
    if len(text) < args.block_size + 2:
        raise ValueError("training text must contain more characters than --block-size")
    vocabulary = list(range(256))
    data = torch.tensor(list(text.encode("latin-1")), dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPTConfig(vocab_size=len(vocabulary), block_size=args.block_size)
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = get_batch(data, args.block_size, args.batch_size, device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f"step {step:5d}/{args.steps} | loss {loss.item():.4f} | device {device}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**model.checkpoint(), "vocabulary": vocabulary, "encoding": "utf-8-bytes"}, args.out)
    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
