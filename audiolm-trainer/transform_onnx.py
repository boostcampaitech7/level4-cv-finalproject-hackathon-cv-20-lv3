import sys
import argparse
import torch.onnx
from pathlib import Path
# Add custom module path
sys.path.append(str(Path(__file__).parent / "audiolm-trainer"))

from models import load_model
from models.salmonn import SALMONN
from config import Config
from dataset import SALMONNDataset


def parse_args():
    parser = argparse.ArgumentParser(description="train parameters")
    parser.add_argument(
        "--cfg-path", type=str, required=True, help="path to configuration file"
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="if True, use dummy model and skip forward/backward",
    )
    parser.add_argument(
        "--local-rank", type=int, help="designate gpu number for parallel process"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(args)

    model = load_model(cfg.config.model)
    model.eval()

    test_dataset = SALMONNDataset(
        cfg.config.datasets.prefix,
        cfg.config.datasets.train_ann_path,
        cfg.config.datasets.whisper_path,
    )
    samples = test_dataset[0]
    onnx_program = torch.onnx.dynamo_export(model, samples)
    onnx_program.save("model.onnx")


if __name__ == "__main__":
    main()
