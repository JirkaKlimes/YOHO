import argparse
from pathlib import Path
import shutil


parser = argparse.ArgumentParser(description="Create a new YOHO training session")
parser.add_argument("name", type=str, help="Name of the session")

args = parser.parse_args()

path = Path("./sessions/").joinpath(args.name)

if path.exists():
    print("Session with this name already exists!")
    quit()

path.mkdir(parents=True)
shutil.copy(Path("./train/config.toml"), path)

for name in ["weights", "stages"]:
    path.joinpath(name).mkdir()
