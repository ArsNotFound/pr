import argparse
import itertools
import random

from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="Input dir", type=Path)
    parser.add_argument("output_dir", help="Output dir name", type=Path)
    parser.add_argument("--ratio", nargs='+', type=float, help="Output ratio", required=True)
    parser.add_argument("--mask_input_dir", help="Mask input dir", type=Path)
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    mask_input_dir: Path | None = args.mask_input_dir
    ratio: list[float] = args.ratio

    if not input_dir.is_dir():
        print(f"Input directory not exists: {args.input_dir}")
        exit(1)

    if mask_input_dir and not mask_input_dir.is_dir():
        print(f"Input mask directory not exists: {args.input_dir}")
        exit(1)

    if output_dir.exists():
        print(f"Output directory already exists: {args.output_dir}")
        exit(1)

    for r in ratio:
        (output_dir / str(r)).mkdir(parents=True)

    if args.mask_input_dir:
        for r in args.ratio:
            (output_dir / (str(r) + "_mask")).mkdir(parents=True)

    file_ids: list[str] = list(
        sorted(
            map(lambda x: x.with_suffix('').name,
                filter(lambda x: x.is_file(),
                       input_dir.glob('[!.]*'))
                )
        )
    )

    if len(file_ids) != len(set(file_ids)):
        print("Input directory contains duplicates!")
        exit(1)

    if mask_input_dir:
        mask_ids = list(
            sorted(
                map(lambda x: x.with_suffix('').name,
                    filter(lambda x: x.is_file(),
                           mask_input_dir.glob('[!.]*'))
                    )
            )
        )

        if len(mask_ids) != len(set(mask_ids)):
            print("Mask directory contains duplicates!")
            exit(1)

        if mask_ids != file_ids:
            print("Masks and files not equal!")
            exit(1)

    random.shuffle(file_ids)

    ratio_list = [int(len(file_ids) * r) for r in ratio]
    it = iter(file_ids)
    split_list = [list(itertools.islice(it, 0, elem)) for elem in ratio_list]

    i = 0
    for elem in it:
        split_list[i].append(elem)
        i += 1
        i %= len(split_list)

    for r, items in zip(ratio, split_list):
        current_dir = output_dir / str(r)
        current_mask_dir = output_dir / (str(r) + "_mask")
        for item in tqdm(items):
            current_file: Path = next(input_dir.glob(f"{item}.*"))
            current_file.rename(current_dir / current_file.name)
            if mask_input_dir:
                current_mask: Path = next(mask_input_dir.glob(f"{item}.*"))
                current_mask.rename(current_mask_dir / current_file.name)


if __name__ == "__main__":
    main()
