"""DU command-line interface."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import uvicorn

from du.api.server import deploy_api
from du.app.dashboard import run_app
from du.core.loader import load
from du.eda.insights import insights
from du.eda.summary import summary
from du.ml.train import train


def _cmd_run(data_path: str) -> None:
    df = load(data_path)
    stats = summary(df)
    print("Shape:", stats["shape"])
    print("Missing values:", stats["missing_values"])
    print("Data types:", stats["dtypes"])
    print("Insights:")
    for item in insights(df):
        print(f"- {item}")


def _cmd_train(data_path: str, target: str, output_model: str | None = None) -> None:
    df = load(data_path)
    result = train(df, target)
    output = output_model or "du_model.pkl"
    with open(output, "wb") as f:
        pickle.dump(result.model, f)
    print(f"Trained {result.task_type} model saved to {output}")


def _cmd_app(data_path: str) -> None:
    df = load(data_path)
    run_app(df)


def _cmd_api(model_path: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    app = deploy_api(model)
    uvicorn.run(app, host=host, port=port)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(prog="du", description="DU Data Understanding CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run auto EDA on a dataset")
    run_parser.add_argument("data", help="Path to CSV/Excel/JSON file")

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("data", help="Path to dataset")
    train_parser.add_argument("target", help="Target column")
    train_parser.add_argument("--output", help="Output model path", default="du_model.pkl")

    app_parser = subparsers.add_parser("app", help="Launch Streamlit app")
    app_parser.add_argument("data", help="Path to dataset")

    api_parser = subparsers.add_parser("api", help="Serve model via FastAPI")
    api_parser.add_argument("model", help="Path to pickled model")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8000)

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args.data)
    elif args.command == "train":
        _cmd_train(args.data, args.target, args.output)
    elif args.command == "app":
        _cmd_app(args.data)
    elif args.command == "api":
        _cmd_api(args.model, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
