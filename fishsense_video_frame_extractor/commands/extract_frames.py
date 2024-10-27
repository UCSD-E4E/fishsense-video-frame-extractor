from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from appdirs import user_cache_dir
from fishsense_common.pluggable_cli import Command
from skimage import metrics

from fishsense_video_frame_extractor import __version__


def __diff_images(a: np.ndarray, b: np.ndarray, skip_count: int) -> bool:
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    b_gray = cv2.resize(
        b_gray, (a_gray.shape[1], a_gray.shape[0]), interpolation=cv2.INTER_AREA
    )
    ssim_score = metrics.structural_similarity(a_gray, b_gray, full=True)
    threshold = 0.85 + float(skip_count) / 2000.0
    print(f"ssim: {ssim_score[0]} < {threshold}")
    return ssim_score[0] < threshold


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

    return len(other_images) > 0 and all(
        __diff_images(img, cv2.imread(i.absolute().as_posix()), 100)
        for i in other_images
    )


def __save_image(img: np.ndarray, file: Path):
    if file.exists():
        return

    print("Saving images")
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


def execute(file: Path, output_directory: Path):
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
    amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Handle recovery
    jpgs = list(output_target_directory.glob("*.jpg"))
    if len(jpgs):
        frame_number = max(int(f.stem) for f in jpgs)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    prev_skip_count = 1
    skip_count = 1
    while cap.isOpened():
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print(
            f"Percent: {float(frame_number) / float(amount_of_frames)}, Skip: {skip_count}"
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number + skip_count - 1)
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


class ExtractFrames(Command):
    @property
    def name(self) -> str:
        return "extract-frames"

    @property
    def description(self) -> str:
        return "Extracts frames from specified videos."

    def __init__(self) -> None:
        super().__init__()

    def __call__(self):
        self.init_ray()

        execute(Path("/home/chris/GX020558.MP4"), Path("./output"))
