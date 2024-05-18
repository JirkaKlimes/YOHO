from abc import ABC, abstractmethod
from typing import Any
import multiprocessing as mp
import warnings
import numpy as np


class Dataloader(ABC):
    @abstractmethod
    def __init__(
        self,
        batch_size: int,
        max_queued_batches: int = 8,
        num_workers: int = 4,
        warmup_queue: bool = True,
    ):
        self.batch_size = batch_size
        self.max_queued_batches = max_queued_batches
        self.num_workers = num_workers
        self.warmup_queue = warmup_queue

        self._batch_queue = mp.Queue(self.max_queued_batches)
        self._task_queues = [mp.Queue() for _ in range(num_workers)]

        self.start_workers()

    @abstractmethod
    def get_num_batches(self) -> int:
        """Return number of batches in the dataset"""

    @abstractmethod
    def get_batch(self, idx: int) -> Any:
        """Return batch at specified index"""

    @property
    def num_batches(self) -> int:
        return self.get_num_batches()

    @property
    def num_prepared_batches(self) -> int:
        return self._batch_queue.qsize()

    def _worker(self, idx: int):
        task_queue = self._task_queues[idx]
        while True:
            batch_idx = task_queue.get()
            if batch_idx is None:
                quit()
            batch = self.get_batch(batch_idx)
            self._batch_queue.put(batch)

    def start_workers(self):
        self.current_batch_idx = 0

        for _ in range(self.max_queued_batches):
            self._task_queues[self.current_batch_idx % self.num_workers].put(self.current_batch_idx)
            self.current_batch_idx = (self.current_batch_idx + 1) % self.num_batches

        self._workers = [
            mp.Process(target=self._worker, args=(i,), daemon=True) for i in range(self.num_workers)
        ]
        for worker in self._workers:
            worker.start()

        while self.warmup_queue and not self._batch_queue.full():
            print("warming up")
            continue

    def get_prepared_batch(self):
        if self.num_prepared_batches == 0:
            warnings.warn(
                f"Batches aren't preparing fast enough. Consider optimizing `{self.__class__.__name__}.{self.get_batch.__name__}` method"
            )
        batch = self._batch_queue.get()
        self._task_queues[self.current_batch_idx % self.num_workers].put(self.current_batch_idx)
        self.current_batch_idx = (self.current_batch_idx + 1) % self.num_batches
        return batch

    def __del__(self):
        for q in self._task_queues:
            q.put(None)

        while not self._batch_queue.empty():
            self._batch_queue.get()

        for worker in self._workers:
            worker.join()

    terminate = __del__


if __name__ == "__main__":
    import time

    class CustomDataloader(Dataloader):
        def __init__(
            self,
            batch_size: int,
            max_queue_size: int = 8,
            max_processes: int = 4,
            warmup_queue: bool = True,
        ):
            self.data_x = np.random.normal(size=(1024, 16))
            self.data_y = np.random.normal(size=(1024, 4))
            super().__init__(batch_size, max_queue_size, max_processes, warmup_queue)

        def get_num_batches(self):
            return self.data_x.shape[0] // self.batch_size

        def get_batch(self, idx: int) -> Any:
            i = idx * self.batch_size
            j = (idx + 1) * self.batch_size

            batch_x = self.data_x[i:j]
            batch_y = self.data_y[i:j]

            # some CPU intensive pre-processing like augmentation
            for _ in range(32):
                batch_x += np.random.normal(size=batch_x.shape)

            return batch_x, batch_y

    dataloader = CustomDataloader(32, 16, 8)

    STEPS_PER_EPOCH = 5
    for _ in range(STEPS_PER_EPOCH):
        x, y = dataloader.get_prepared_batch()

        # Some training logic
        time.sleep(0.1)

    dataloader.terminate()
