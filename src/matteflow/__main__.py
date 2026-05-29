"""MatteFlow package CLI entrypoint."""

from .cli_app import _resolve_output_dir, build_parser, main


if __name__ == "__main__":
    raise SystemExit(main())
