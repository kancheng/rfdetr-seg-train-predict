from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import supervision as sv
import torch
from PIL import Image
from rfdetr import RFDETRSegMedium, RFDETRSegNano, RFDETRSegSmall


@dataclass
class Paths:
    project_root: Path = Path("F:/detr")
    yolo_dataset_dir: Path = Path("F:/detr/sdsaliency900/sdsaliency900_dataset")
    predict_input_dir: Path = Path("F:/detr/sdsaliency900/dataset_predict")
    coco_dataset_dir: Path = Path("F:/detr/sdsaliency900/rfdetr_coco_dataset")
    train_output_dir: Path = Path("F:/detr/sdsaliency900/rfdetr_train_output")
    predict_output_dir: Path = Path("F:/detr/sdsaliency900/predict_results")


TARGET_IMAGE_SIZE = (216, 216)


def _stage_log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [PIPELINE] {message}")


def _ensure_model_weights_on_device(
    rfdetr: RFDETRSegMedium | RFDETRSegNano | RFDETRSegSmall,
) -> None:
    """
    trainer.fit 結束後 Lightning 可能把 nn.Module 留在 CPU，但 predict() 會依
    ModelContext.device 把輸入送到 GPU，導致 conv 權重與輸入裝置不一致。
    推論前將權重移回與 device 一致。
    """
    ctx = rfdetr.model
    ctx.model = ctx.model.to(ctx.device)
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()


def _polygon_area(points: list[float], width: int, height: int) -> float:
    coords = []
    for i in range(0, len(points), 2):
        x = points[i] * width
        y = points[i + 1] * height
        coords.append((x, y))

    if len(coords) < 3:
        return 0.0

    area = 0.0
    n = len(coords)
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _bbox_from_polygon(points: list[float], width: int, height: int) -> list[float]:
    xs = [points[i] * width for i in range(0, len(points), 2)]
    ys = [points[i + 1] * height for i in range(0, len(points), 2)]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def convert_yolo_seg_to_coco(paths: Paths, split: str) -> None:
    src_images = paths.yolo_dataset_dir / "images" / split
    src_labels = paths.yolo_dataset_dir / "labels" / split
    dst_split = paths.coco_dataset_dir / split
    dst_split.mkdir(parents=True, exist_ok=True)

    images_json: list[dict[str, Any]] = []
    annotations_json: list[dict[str, Any]] = []
    categories_json = [{"id": 1, "name": "object", "supercategory": "object"}]

    annotation_id = 1
    image_id = 1

    for image_path in sorted(src_images.glob("*.png")):
        label_path = src_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        with Image.open(image_path) as img:
            resized = img.convert("RGB").resize(TARGET_IMAGE_SIZE, Image.BILINEAR)
            width, height = resized.size
            resized.save(dst_split / image_path.name)

        images_json.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            class_id = int(float(parts[0]))
            polygon = [float(v) for v in parts[1:]]

            if len(polygon) % 2 != 0:
                continue

            area = _polygon_area(polygon, width, height)
            bbox = _bbox_from_polygon(polygon, width, height)

            segmentation_px = []
            for i in range(0, len(polygon), 2):
                segmentation_px.extend([polygon[i] * width, polygon[i + 1] * height])

            annotations_json.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "segmentation": [segmentation_px],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        image_id += 1

    coco_json = {
        "images": images_json,
        "annotations": annotations_json,
        "categories": categories_json,
    }
    (dst_split / "_annotations.coco.json").write_text(
        json.dumps(coco_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare_coco_dataset(paths: Paths) -> None:
    _stage_log("開始準備 COCO 資料集")
    started_at = time.perf_counter()
    if paths.coco_dataset_dir.exists():
        shutil.rmtree(paths.coco_dataset_dir)
    (paths.coco_dataset_dir / "train").mkdir(parents=True, exist_ok=True)
    (paths.coco_dataset_dir / "valid").mkdir(parents=True, exist_ok=True)

    convert_yolo_seg_to_coco(paths, "train")
    convert_yolo_seg_to_coco(paths, "val")
    # RF-DETR custom dataset loader expects "valid" split naming.
    val_dir = paths.coco_dataset_dir / "val"
    valid_dir = paths.coco_dataset_dir / "valid"
    if val_dir.exists():
        for item in val_dir.iterdir():
            target = valid_dir / item.name
            if target.exists():
                if target.is_file():
                    target.unlink()
                else:
                    shutil.rmtree(target)
            if item.is_file():
                shutil.copy2(item, target)
            else:
                shutil.copytree(item, target)
        shutil.rmtree(val_dir)
    elapsed = time.perf_counter() - started_at
    _stage_log(f"COCO 資料集準備完成，耗時 {elapsed:.1f} 秒，輸出: {paths.coco_dataset_dir}")


def _build_model(model_size: str, pretrain_weights: str | None = None) -> RFDETRSegMedium:
    model_kwargs: dict[str, Any] = {}
    if pretrain_weights:
        model_kwargs["pretrain_weights"] = pretrain_weights

    if model_size == "nano":
        return RFDETRSegNano(**model_kwargs)
    if model_size == "small":
        return RFDETRSegSmall(**model_kwargs)
    return RFDETRSegMedium(**model_kwargs)


def _build_model_with_retry(
    model_size: str, max_retries: int = 5, base_sleep_seconds: int = 5
) -> RFDETRSegMedium:
    """
    Build RF-DETR model with retry for transient download/network failures.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return _build_model(model_size=model_size)
        except (
            requests.exceptions.RequestException,
            requests.exceptions.ChunkedEncodingError,
            OSError,
        ) as exc:
            if attempt == max_retries:
                raise
            wait_seconds = base_sleep_seconds * attempt
            print(
                f"[警告] 權重下載/初始化失敗 (第 {attempt}/{max_retries} 次): {exc}\n"
                f"將於 {wait_seconds} 秒後重試..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError("無法初始化 RF-DETR 模型。")


def train_model(
    paths: Paths, model_size: str, epochs: int = 50
) -> RFDETRSegMedium | RFDETRSegNano | RFDETRSegSmall:
    _stage_log("開始初始化訓練模型")
    model = _build_model_with_retry(model_size=model_size, max_retries=5, base_sleep_seconds=8)
    _stage_log("模型初始化完成，開始訓練流程")
    started_at = time.perf_counter()
    for attempt in range(1, 4):
        try:
            _stage_log(f"開始訓練（第 {attempt} 次嘗試），epochs={epochs}")
            model.train(
                dataset_dir=str(paths.coco_dataset_dir),
                epochs=epochs,
                output_dir=str(paths.train_output_dir),
                batch_size=1,
                grad_accum_steps=4,
            )
            elapsed = time.perf_counter() - started_at
            _stage_log(f"訓練完成，耗時 {elapsed / 60:.1f} 分鐘，輸出: {paths.train_output_dir}")
            break
        except (requests.exceptions.RequestException, OSError) as exc:
            if attempt == 3:
                raise
            wait_seconds = 10 * attempt
            print(
                f"[警告] 訓練啟動失敗 (第 {attempt}/3 次): {exc}\n"
                f"將於 {wait_seconds} 秒後重試訓練..."
            )
            time.sleep(wait_seconds)
    return model


def _pick_latest_weight_file(search_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for pattern in ("*.pt", "*.pth", "*.ckpt"):
        candidates.extend(search_dir.rglob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_model_for_predict_only(
    paths: Paths, predict_weights: str | None, model_size: str
) -> RFDETRSegMedium | RFDETRSegNano | RFDETRSegSmall:
    if predict_weights:
        weight_path = Path(predict_weights)
        if not weight_path.exists():
            raise FileNotFoundError(f"指定的權重檔不存在: {weight_path}")
        print(f"[INFO] predict_only 模式使用指定權重: {weight_path}")
        return _build_model(model_size=model_size, pretrain_weights=str(weight_path))

    latest = _pick_latest_weight_file(paths.train_output_dir)
    if latest:
        print(f"[INFO] predict_only 模式自動使用最新權重: {latest}")
        return _build_model(model_size=model_size, pretrain_weights=str(latest))

    print("[警告] 找不到自訓權重，將改用官方預訓練權重進行推論。")
    return _build_model_with_retry(model_size=model_size, max_retries=5, base_sleep_seconds=8)


def run_prediction(
    model: RFDETRSegMedium | RFDETRSegNano | RFDETRSegSmall,
    paths: Paths,
    threshold: float = 0.5,
) -> None:
    _ensure_model_weights_on_device(model)
    _stage_log("開始批次推論")
    started_at = time.perf_counter()
    paths.predict_output_dir.mkdir(parents=True, exist_ok=True)
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()

    image_paths = sorted(paths.predict_input_dir.glob("*.png"))
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        detections = model.predict(image, threshold=threshold)

        labels = []
        if detections.class_id is not None:
            labels = [f"class_{int(class_id)}" for class_id in detections.class_id]

        annotated = mask_annotator.annotate(image.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels)
        if isinstance(annotated, Image.Image):
            annotated.save(paths.predict_output_dir / image_path.name)
        else:
            Image.fromarray(annotated).save(paths.predict_output_dir / image_path.name)
        if index % 100 == 0 or index == total:
            _stage_log(f"推論進度: {index}/{total}")
    elapsed = time.perf_counter() - started_at
    _stage_log(f"批次推論完成，耗時 {elapsed / 60:.1f} 分鐘")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RF-DETR segmentation train/predict pipeline")
    parser.add_argument(
        "--mode",
        choices=["train_predict", "predict_only"],
        default="train_predict",
        help="train_predict: 先訓練再推論；predict_only: 只推論",
    )
    parser.add_argument(
        "--predict-weights",
        type=str,
        default=None,
        help="predict_only 模式下指定權重檔路徑（.pt/.pth/.ckpt）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="推論信心閾值",
    )
    parser.add_argument(
        "--model-size",
        choices=["nano", "small", "medium"],
        default="nano",
        help="模型大小（4GB 顯存建議 nano）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="訓練輪數（僅 train_predict 有效，預設 50）",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs 必須為 >= 1 的整數")
    return args


def main() -> None:
    total_started_at = time.perf_counter()
    args = parse_args()
    paths = Paths()
    _stage_log(
        f"程式啟動，模式: {args.mode}，模型: {args.model_size}"
        + (f"，epochs: {args.epochs}" if args.mode == "train_predict" else "")
    )

    if args.mode == "train_predict":
        prepare_coco_dataset(paths)
        model = train_model(paths, model_size=args.model_size, epochs=args.epochs)
    else:
        _stage_log("跳過訓練，使用 predict_only 模式")
        model = build_model_for_predict_only(paths, args.predict_weights, model_size=args.model_size)

    run_prediction(model, paths, threshold=args.threshold)
    total_elapsed = time.perf_counter() - total_started_at
    _stage_log(f"流程完成，總耗時 {total_elapsed / 60:.1f} 分鐘")
    print(f"推論結果已輸出到: {paths.predict_output_dir}")


if __name__ == "__main__":
    main()