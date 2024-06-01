import argparse
import os

from train.utils.config import load_config


parser = argparse.ArgumentParser(
    description="Starts/Resumes the training of session",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument("name", type=str, help="Name of the session")

parser.add_argument(
    "stage",
    type=int,
    choices=[0, 1, 2, 3, 4],
    help=(
        "Stage of training to start/resume:\n"
        "\t0 - Train tokenizer\n"
        "\t1 - Pretrain transcription without voiceprints\n"
        "\t2 - Train voice reconstruction\n"
        "\t3 - Finetune voiceprint encoder\n"
        "\t4 - Finetune transcription with voiceprints"
    ),
)

args = parser.parse_args()
config = load_config(args.name)

if config.hardware.devices != "all":
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.devices))
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(config.hardware.allowed_mem_fraction)


match args.stage:
    case 0:
        print("Loaded config:")
        print(config.model_dump_json(indent=4))

        from train.stages.train_tokenizer import main

        main(config)

    case 1:
        print("Loaded config:")
        print(config.model_dump_json(indent=4))

        from train.stages.transcription_pretrain_no_voiceprints import main

        main(config)

    case _:
        raise NotImplementedError("Stage has not been implemented yet...")
