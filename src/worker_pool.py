import multiprocessing
import logging
from queue import Empty
from typing import Callable, Any


class WorkerPool:
    def __init__(self, worker_function: Callable[[Any], None], num_workers: int = 1):
        """
        Initialize a worker pool that processes tasks using multiple processes.

        Args:
            worker_function: Function that will be called with each work item
            num_workers: Number of worker processes to spawn
        """
        self.worker_function = worker_function
        self.num_workers = num_workers
        self.queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.processes: list[multiprocessing.Process] = []

        for _ in range(self.num_workers):
            process = multiprocessing.Process(target=self._worker)
            process.start()
            self.processes.append(process)

    def _worker(self):
        """Worker process that continuously pulls and processes items from the queue."""
        while True:
            if self.stop_event.is_set():
                break
            try:
                work_item = self.queue.get(timeout=1)
            except (Empty, BrokenPipeError, EOFError, ConnectionResetError):
                # Handle various connection/pipe issues gracefully
                continue
            except Exception as e:
                # Log unexpected queue errors but continue
                logging.warning(f'Unexpected queue error: {e}')
                continue

            try:
                self.worker_function(work_item)
            except Exception as e:
                logging.error(f'Worker function failed for {work_item}: {e}')

    def submit(self, work_item: Any) -> None:
        """Submit a work item to be processed by the worker pool."""
        try:
            self.queue.put(work_item)
        except Exception as e:
            logging.warning(f'Unexpected queue submission error: {e}')

    def stop(self):
        """Stop all worker processes and wait for them to finish."""
        self.stop_event.set()
        for process in self.processes:
            process.join()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically stop workers."""
        self.stop()
