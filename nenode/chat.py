import argparse
from pathlib import Path

import torch

from .model import GPT, GPTConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a trained Nenode model.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/nenode.pt"))
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("encoding") != "utf-8-bytes":
        raise ValueError("checkpoint was created by an older incompatible trainer; train a new checkpoint")
    model = GPT(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print("Nenode is ready. Type /quit to exit.")

    while True:
        try:
            prompt = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip() == "/quit":
            break
        tokens = torch.tensor([[token for token in prompt.encode("utf-8")]], device=device)
        generated = model.generate(tokens, args.tokens, args.temperature)[0].tolist()
        print("ai> " + bytes(generated).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
