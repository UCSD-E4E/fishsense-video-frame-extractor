import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import ray
from appdirs import user_cache_dir
from fishsense_common.pluggable_cli import Command
from skimage import metrics
from tqdm import tqdm

from fishsense_video_frame_extractor import __version__


@ray.remote(num_cpus=0)
class ProgressReporter:
    def __init__(self):
        self.updated = asyncio.Event()
        self.value: Tuple[Path, int] = None

    async def get_update_blocking(self) -> Tuple[Path, int]:
        await self.updated.wait()
        self.updated.clear()
        file, frame = self.value
        self.value = None

        return file, frame

    def update(self, file: Path, frame: int):
        self.value = (file, frame)
        self.updated.set()


def __diff_images(a: np.ndarray, b: np.ndarray, skip_count: int) -> bool:
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    b_gray = cv2.resize(
        b_gray, (a_gray.shape[1], a_gray.shape[0]), interpolation=cv2.INTER_AREA
    )
    ssim_score = metrics.structural_similarity(a_gray, b_gray, multichannel=True)
    threshold = 0.85 + float(skip_count) / 2000.0
    return ssim_score < threshold


def __is_unique_image(
    img: np.ndarray, output_directory: Path, target_directory: Path, skip_count: int
) -> bool:
    # By reversing this, we check the image that is going to be most similar first.
    # This allows us to trend towards only checking a single image most of the time.
    other_target_images = list(target_directory.glob("*.jpg"))
    other_target_images.sort(key=lambda x: int(x.stem), reverse=True)

    if any(
        not __diff_images(img, cv2.imread(i.absolute().as_posix()), skip_count)
        for i in other_target_images
    ):
        return False

    # We don't need to test any of the above images.
    other_images = set(output_directory.glob("**/*.jpg"))
    other_images.difference_update(other_target_images)

    if len(other_target_images) == 0 and len(other_images) == 0:
        return True

    return all(
        __diff_images(img, cv2.imread(i.absolute().as_posix()), 100)
        for i in other_images
    )


def __save_image(img: np.ndarray, file: Path):
    if file.exists():
        return

    cv2.imwrite(file.absolute().as_posix(), img)


def __store_new_image(
    img: np.ndarray,
    file: Path,
    output_directory: Path,
    target_directory: Path,
    skip_count: int,
) -> Tuple[np.ndarray, bool]:
    if __is_unique_image(img, output_directory, target_directory, skip_count):
        __save_image(img, file)
        return True

    return False


@ray.remote
def execute(file: Path, output_directory: Path, reporter: ProgressReporter):
    cache_dir = (
        Path(
            user_cache_dir(
                "fishsense_video_frame_extractor",
                appauthor="Engineers for Exploration",
                version=__version__,
            )
        )
        / "google_drive"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_target_directory = output_directory / file.stem
    output_target_directory.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(file.absolute().as_posix())

    # Handle recovery
    jpgs = list(output_target_directory.glob("*.jpg"))
    if len(jpgs):
        frame_number = max(int(f.stem) for f in jpgs)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    prev_skip_count = 1
    skip_count = 1
    while cap.isOpened():
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number + skip_count - 1)
        reporter.update.remote(file, int(frame_number))
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            break

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        stored_new_image = __store_new_image(
            frame,
            output_target_directory / f"{int(frame_number)}.jpg",
            output_directory,
            output_target_directory,
            skip_count,
        )

        # Fibonacci Seek
        if not stored_new_image:
            temp = skip_count
            skip_count += prev_skip_count
            prev_skip_count = temp
        else:
            skip_count = 1
            prev_skip_count = 1

    cap.release()


def print_progress(
    reporter: ProgressReporter, files: List[Path], frame_counts: List[int]
):
    total_frame_count = sum(frame_counts)
    total_tqdm = tqdm(total=total_frame_count, position=1, desc="Total Frames")
    prev_total_completed = 0

    file_frame_number = {}
    file_totals = {f: c for f, c in zip(files, frame_counts)}
    file_tqdm: Dict[Path, tqdm] = {}
    file_prev_completed: Dict[Path, int] = {}

    while True:
        file, frame_number = ray.get(reporter.get_update_blocking.remote())

        file_frame_number[file] = frame_number
        total_completed = sum(v for _, v in file_frame_number.items())
        total_tqdm.update(total_completed - prev_total_completed)
        total_tqdm.refresh()
        prev_total_completed = total_completed

        if file not in file_tqdm:
            file_tqdm[file] = tqdm(
                total=file_totals[file],
                position=len(file_tqdm) + 2,
                desc=file.stem,
                leave=False,
            )
            file_prev_completed[file] = 0

        file_tqdm[file].update(frame_number - file_prev_completed[file])
        file_tqdm[file].refresh()
        file_prev_completed[file] = frame_number


class ExtractFrames(Command):
    @property
    def name(self) -> str:
        return "extract-frames"

    @property
    def description(self) -> str:
        return "Extracts frames from specified videos."

    def __init__(self) -> None:
        super().__init__()

    def __get_frame_count(self, file: Path) -> int:
        cap = cv2.VideoCapture(file.absolute().as_posix())
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        return frame_count

    def __call__(self):
        self.init_ray()

        files = ["/home/chris/GX010021.MP4", "/home/chris/GX020558.MP4"]
        files = [Path(f) for f in files]

        output = Path("./output")

        frame_counts = [self.__get_frame_count(f) for f in files]
        reporter: ProgressReporter = ProgressReporter.remote()

        reporter_thread = threading.Thread(
            target=print_progress, args=(reporter, files, frame_counts)
        )
        reporter_thread.start()

        futures = [execute.remote(f, output, reporter) for f in files]
        list(self.tqdm(futures, total=len(files), desc="Total Videos"))
