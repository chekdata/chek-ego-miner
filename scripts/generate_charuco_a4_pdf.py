#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image


DICTIONARY_NAME_TO_ID = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
}

A4_PORTRAIT_MM = (210.0, 297.0)
A4_LANDSCAPE_MM = (297.0, 210.0)


def mm_to_px(mm: float, dpi: int) -> int:
    return round(mm / 25.4 * dpi)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an A4 Charuco board PDF for printing.")
    parser.add_argument("--output-dir", required=True, help="Directory for the generated PDF/PNG.")
    parser.add_argument("--board-squares-x", type=int, default=8)
    parser.add_argument("--board-squares-y", type=int, default=6)
    parser.add_argument("--square-size-mm", type=float, default=24.0)
    parser.add_argument("--marker-size-mm", type=float, default=18.0)
    parser.add_argument("--dictionary-name", default="DICT_4X4_50", choices=sorted(DICTIONARY_NAME_TO_ID.keys()))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--landscape", action="store_true", help="Generate an A4 landscape sheet.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.marker_size_mm >= args.square_size_mm:
        raise SystemExit("marker-size-mm must be smaller than square-size-mm")

    page_w_mm, page_h_mm = A4_LANDSCAPE_MM if args.landscape else A4_PORTRAIT_MM
    board_w_mm = args.board_squares_x * args.square_size_mm
    board_h_mm = args.board_squares_y * args.square_size_mm
    if board_w_mm > page_w_mm or board_h_mm > page_h_mm:
        raise SystemExit(
            f"Board {board_w_mm:.1f}mm x {board_h_mm:.1f}mm does not fit on "
            f"{page_w_mm:.1f}mm x {page_h_mm:.1f}mm A4."
        )

    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_NAME_TO_ID[args.dictionary_name])
    board = cv2.aruco.CharucoBoard(
        (args.board_squares_x, args.board_squares_y),
        args.square_size_mm / 1000.0,
        args.marker_size_mm / 1000.0,
        dictionary,
    )

    page_w_px = mm_to_px(page_w_mm, args.dpi)
    page_h_px = mm_to_px(page_h_mm, args.dpi)
    board_w_px = mm_to_px(board_w_mm, args.dpi)
    board_h_px = mm_to_px(board_h_mm, args.dpi)
    offset_x = (page_w_px - board_w_px) // 2
    offset_y = (page_h_px - board_h_px) // 2

    board_img = board.generateImage((board_w_px, board_h_px))
    page_img = Image.new("L", (page_w_px, page_h_px), color=255)
    page_img.paste(Image.fromarray(board_img), (offset_x, offset_y))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    orientation = "landscape" if args.landscape else "portrait"
    stem = (
        f"charuco_a4_{orientation}_{args.board_squares_x}x{args.board_squares_y}"
        f"_{int(args.square_size_mm)}mm_{int(args.marker_size_mm)}mm_{args.dictionary_name.lower()}"
    )
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    page_img.save(png_path, dpi=(args.dpi, args.dpi))
    page_img.convert("RGB").save(pdf_path, resolution=args.dpi)

    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(
        "Use these calibration values: "
        f"squares_x={args.board_squares_x}, squares_y={args.board_squares_y}, "
        f"square_size_mm={args.square_size_mm}, marker_size_mm={args.marker_size_mm}, "
        f"dictionary={args.dictionary_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
