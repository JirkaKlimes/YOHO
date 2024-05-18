from abc import ABC, abstractmethod
from typing import Any
import multiprocessing as mp
import warnings


class Dataloader(ABC):
    @abstractmethod
    def __init__(
        self,
        batch_size: int,
        max_queued_batches: int = 8,
        num_workers: int = 4,
        warmup_queue: bool = True,
        use_multiprocessing: bool = True,
    ):
        self.batch_size = batch_size
        self.max_queued_batches = max_queued_batches
        self.num_workers = num_workers
        self.warmup_queue = warmup_queue
        self.use_multiprocessing = use_multiprocessing

        self.current_batch_idx = 0

        if self.use_multiprocessing:
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
        for _ in range(self.max_queued_batches):
            self._task_queues[self.current_batch_idx % self.num_workers].put(self.current_batch_idx)
            self.current_batch_idx = (self.current_batch_idx + 1) % self.num_batches

        self._workers = [
            mp.Process(target=self._worker, args=(i,), daemon=True) for i in range(self.num_workers)
        ]
        for worker in self._workers:
            worker.start()

        while self.warmup_queue and not self._batch_queue.full():
            continue

    def get_prepared_batch(self):
        if not self.use_multiprocessing:
            batch = self.get_batch(self.current_batch_idx)
            self.current_batch_idx = (self.current_batch_idx + 1) % self.num_batches
            return batch

        if self.num_prepared_batches == 0:
            warnings.warn(
                f"Batches aren't preparing fast enough. Consider optimizing `{self.__class__.__name__}.{self.get_batch.__name__}` method"
            )
        batch = self._batch_queue.get()
        self._task_queues[self.current_batch_idx % self.num_workers].put(self.current_batch_idx)
        self.current_batch_idx = (self.current_batch_idx + 1) % self.num_batches
        return batch

    def terminate(self):
        for q in self._task_queues:
            q.put(None)

        while not self._batch_queue.empty():
            self._batch_queue.get()

        for worker in self._workers:
            worker.join()
