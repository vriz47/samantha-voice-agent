import queue
import subprocess
import threading


class AudioQueue:
    """Play generated audio files back to back on a single player process.

    ``termux-media-player`` accepts only one file per invocation, so playback is
    serialised on a worker thread. Producers enqueue a file as soon as it is
    written and return immediately, which lets TTS keep synthesising the next
    clause while the previous one is already playing.
    """

    def __init__(self, command: list[str], timeout: float = 300.0):
        self._cmd = command
        self._timeout = timeout
        self._items: queue.Queue[str | None] = queue.Queue()
        self._error: BaseException | None = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            path = self._items.get()
            if path is None:
                self._items.task_done()
                return
            try:
                subprocess.run(
                    self._cmd + [path], capture_output=True, timeout=self._timeout
                )
            except (subprocess.SubprocessError, OSError) as exc:
                self._error = exc
            finally:
                self._items.task_done()

    @property
    def error(self) -> BaseException | None:
        return self._error

    def put(self, path: str) -> None:
        self._items.put(path)

    def drain(self) -> None:
        """Block until every queued file has finished playing."""
        self._items.join()

    def close(self) -> None:
        self._items.put(None)