import asyncio
import threading
from glob import glob
from pathlib import Path
from random import sample, seed
from shutil import copy
from typing import Dict, List, Tuple

import cv2
import numpy as np
import ray
from appdirs import user_cache_dir
from fishsense_common.pluggable_cli import Command, argument
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
    b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    ssim_score = metrics.structural_similarity(a, b, channel_axis=2)
    threshold = 0.7 + float(skip_count) / 1500.0
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

    file.parent.mkdir(parents=True, exist_ok=True)

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
def execute(
    file: Path, output_directory: Path, root_directory: Path, reporter: ProgressReporter
):
    output_target_directory = (
        Path(
            file.parent.absolute()
            .as_posix()
            .replace(root_directory.as_posix(), output_directory.absolute().as_posix())
        )
        / file.stem
    )
    output_target_directory.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(file.absolute().as_posix())
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Handle recovery
    jpgs = list(output_target_directory.glob("*.jpg"))
    if len(jpgs):
        frame_number = max(int(f.stem) for f in jpgs)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    prev_skip_count = 1
    skip_count = 1
    while cap.isOpened():
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        frame_number += skip_count - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
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

    reporter.update.remote(file, int(frame_count))


def print_progress(
    reporter: ProgressReporter, files: List[Path], frame_counts: List[int]
):
    videos_completed = 0

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
                leave=True,
            )
            file_prev_completed[file] = 0

        file_tqdm[file].update(frame_number - file_prev_completed[file])
        file_tqdm[file].refresh()
        file_prev_completed[file] = frame_number

        # Exit out when all videos are completed so the process ends.
        if frame_number == file_totals[file]:
            videos_completed += 1

            if videos_completed == len(files):
                break


class ExtractFrames(Command):
    @property
    def name(self) -> str:
        return "extract-frames"

    @property
    def description(self) -> str:
        return "Extracts frames from specified videos."

    @property
    @argument("data", required=True, help="A glob that represents the data to process.")
    def data(self) -> List[str]:
        return self.__data

    @data.setter
    def data(self, value: List[str]):
        self.__data = value

    @property
    @argument(
        "--output",
        short_name="-o",
        required=True,
        help="The path to store the resulting database.",
    )
    def output_path(self) -> str:
        return self.__output_path

    @output_path.setter
    def output_path(self, value: str):
        self.__output_path = value

    @property
    @argument(
        "--count",
        short_name="-c",
        default=None,
        help="The total number of videos to choose.  These videos are choosen from the files returned from the glob using a seeded random sample.",
    )
    def count(self) -> int:
        return self.__count

    @count.setter
    def count(self, value: int):
        self.__count = value

    @property
    @argument(
        "--seed",
        short_name="-s",
        default=1234,
        help="The seed to use when choosing videos at random.",
    )
    def seed(self) -> int:
        return self.__seed

    @seed.setter
    def seed(self, value: int):
        self.__seed = value

    def __init__(self) -> None:
        super().__init__()

        self.__data: List[str] = None
        self.__output_path: str = None
        self.__count: int = None
        self.__seed: int = None

    def __cache_video(self, file: Path, root_directory: Path) -> Path:
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
        cache_dir = Path(
            file.parent.absolute()
            .as_posix()
            .replace(root_directory.as_posix(), cache_dir.absolute().as_posix())
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_dir / file.name

        if not cache_file.exists():
            copy(file.absolute().as_posix(), cache_file.absolute().as_posix())

        return cache_file

    def __get_frame_count(self, file: Path) -> int:
        cap = cv2.VideoCapture(file.absolute().as_posix())
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        return frame_count

    def __get_root(self, files: List[Path]) -> Path:
        # Find the singular path that defines the root of all of our data.
        root = files
        while len(root) > 1:
            max_count = max(len(f.parts) for f in root)
            root = {f.parent if len(f.parts) == max_count else f for f in root}
        return root.pop()

    def __call__(self):
        self.init_ray()

        seed(self.seed)

        files = [Path(f) for g in self.data for f in glob(g, recursive=True)]
        output = Path(self.output_path)

        # Find the singular path that defines the root of all of our data.
        root = self.__get_root(files)

        if self.count is not None:
            # Choose whichever we have fewer of
            count = min(self.count, len(files))
            files = sample(files, count)

        files = [
            self.__cache_video(f, root)
            for f in tqdm(files, desc="Caching videos locally")
        ]
        # Update the root after switching to cache.
        root = self.__get_root(files)

        frame_counts = [self.__get_frame_count(f) for f in files]
        reporter: ProgressReporter = ProgressReporter.remote()

        # Use a thread instead of another process because all we are doing is reporting anyways.
        # The GIL won't impact us much.
        reporter_thread = threading.Thread(
            target=print_progress, args=(reporter, files, frame_counts)
        )
        reporter_thread.start()

        futures = [execute.remote(f, output, root, reporter) for f in files]
        list(self.tqdm(futures, total=len(files), desc="Total Videos"))

        print("Waiting for all threads to finish before exiting...")
        reporter_thread.join()
