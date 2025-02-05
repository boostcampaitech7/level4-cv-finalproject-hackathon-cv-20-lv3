import sys
import argparse
import torch.onnx
from pathlib import Path
# Add custom module path
sys.path.append("/data/ephemeral/home/syp/level4-cv-finalproject-hackathon-cv-20-lv3/audiolm-trainer")

from models import load_model
from config import Config
from dataset import SALMONNDataset
from salmonn_utils import SALMONNTestDataset, load_preprocessor, load_model
from utils import get_dataloader, prepare_sample



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

    torch_model = load_preprocessor(cfg)
    test_dataset = SALMONNTestDataset(
        cfg.config.datasets.prefix,
        cfg.config.datasets.test_ann_path_asr,
        cfg.config.datasets.whisper_path,
    )
    samples = test_dataset[0]
    torch.onnx.export(
        torch_model, 
        samples, 
        "model.onnx",
        opset_version=17,
        input_names=["spectrogram", "text", "task"],
        output_names=["loss"],
        dynamic_axes={
            "spectrogram": {0: "batch", 1: "time"},
            "text": {0: "batch"}
        }
    )


if __name__ == "__main__":
    main()
