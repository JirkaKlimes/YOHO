from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import mplcyberpunk
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class DataHandler(FileSystemEventHandler):
    def __init__(self, file_path, update_func):
        self.file_path = file_path
        self.update_func = update_func

    def on_modified(self, event):
        if event.src_path == self.file_path:
            self.update_func()


def plot_live(file_path):
    plt.style.use("cyberpunk")

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    plt.subplots_adjust(hspace=0.5)

    def update_plot():
        data = pd.read_csv(file_path)
        val_data = data.dropna(subset=["val_loss"])

        axs[0].clear()
        axs[1].clear()
        axs[2].clear()

        axs[0].plot(data["update"], data["learning_rate"], linewidth=0.6, label="Learning Rate")
        axs[0].set_xlabel("Update")
        axs[0].set_title("Learning Rate")
        mplcyberpunk.add_glow_effects(axs[0])

        axs[1].plot(data["update"], data["loss"], linewidth=0.6, label="Loss")
        axs[1].set_yscale("log")
        axs[1].set_xlabel("Update")
        axs[1].set_title("Loss")
        mplcyberpunk.add_glow_effects(axs[1])

        axs[2].plot(
            val_data["update"], val_data["val_loss"], linewidth=0.6, label="Validation Loss"
        )
        axs[2].set_yscale("log")
        axs[2].set_xlabel("Update")
        axs[2].set_title("Validation Loss")
        mplcyberpunk.add_glow_effects(axs[2])

        plt.draw()

    update_plot()

    event_handler = DataHandler(file_path, update_plot)
    observer = Observer()
    observer.schedule(event_handler, path=file_path, recursive=False)
    observer.start()

    try:
        plt.show()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Shows live plot of training session",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("name", type=str, help="Name of the session")

    parser.add_argument(
        "stage",
        type=int,
        choices=[0, 1, 2, 3, 4],
        help=(
            "Stage of training to plot:\n"
            "\t0 - Train tokenizer\n"
            "\t1 - Pretrain transcription without voiceprints\n"
            "\t2 - Train voice reconstruction\n"
            "\t3 - Finetune voiceprint encoder\n"
            "\t4 - Finetune transcription with voiceprints"
        ),
    )

    args = parser.parse_args()

    file_path = Path(f"./sessions/{args.name}/stages/{args.stage}/metrics.csv")
    plot_live(file_path)
