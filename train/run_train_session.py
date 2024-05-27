import argparse
from pathlib import Path
import toml

from train.utils.config import SessionConfig


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

path = Path("./sessions/").joinpath(args.name)

if not path.exists():
    print(f"Session {args.name} doesn't exist!")
    quit()


config = SessionConfig(
    name=args.name,
    **toml.load(path.joinpath("config.toml")),
)

for attribute in config.weights.__annotations__.keys():
    current_path = getattr(config.weights, attribute)
    new_path = config.path.joinpath(current_path)
    setattr(config.weights, attribute, new_path)


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
