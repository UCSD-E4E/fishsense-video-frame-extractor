from fishsense_common.pluggable_cli import Cli

from fishsense_video_frame_extractor.commands.extract_frames import ExtractFrames


def main():
    cli = Cli(
        name="fs-frame-extractor",
        description="This command line tool takes in multiple videos and pulls out unqiue frames.",
    )

    cli.add(ExtractFrames())

    cli()


if __name__ == "__main__":
    main()
