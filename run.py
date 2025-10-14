from app import create_app
import sys


def main():
    # CLI subcommand: python run.py multiproc --video /path/to/video.mp4 --processes 10
    if len(sys.argv) > 1 and sys.argv[1] == "multiproc":
        from scripts.multiproc_frame_ranges import cli_main
        cli_main(sys.argv[2:])
        return

    # Default: start the web server
    create_app("development").run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()


