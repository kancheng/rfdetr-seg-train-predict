# RF-DETR 分割訓練與推論使用說明

[English README](./README.md)

1. 將 YOLO segmentation 資料轉為 COCO segmentation 格式  
2. 使用 RF-DETR 進行分割訓練（預設 50 epochs）  
3. 對指定圖片資料夾批次推論，並輸出標註結果圖

## Demo 預測結果

![Demo Predict](./demo_predict.png)

---

## 1. 專案目錄

- `main.py`：主程式（訓練 + 推論）
- `requirements.txt`：基本需求套件
- `requirements-lock.txt`：鎖版本需求套件（建議重現環境用）
- `sdsaliency900/sdsaliency900_dataset`：訓練資料（YOLO segmentation）
- `sdsaliency900/dataset_predict`：推論輸入圖片
- `sdsaliency900/rfdetr_coco_dataset`：程式自動轉出的 COCO 資料
- `sdsaliency900/rfdetr_train_output`：訓練輸出（權重、log）
- `sdsaliency900/predict_results`：推論結果圖片

---

## 2. 環境安裝

建議在 `initpy312` conda 環境中執行。

### 方式 A：一般安裝

```bash
pip install -r requirements.txt
```

### 方式 B：鎖版本安裝（建議）

```bash
pip install -r requirements-lock.txt
```

---

## 3. 執行方式

### 3.1 訓練後推論（預設模式）

```bash
python main.py --mode train_predict
```

說明：
- 先將 YOLO 分割資料轉為 COCO 格式
- 訓練 RF-DETR Seg 模型（50 epochs）
- 再對 `dataset_predict` 進行批次推論

### 3.2 模型大小（預設 `nano`）

```bash
python main.py --mode train_predict --model-size nano
python main.py --mode train_predict --model-size small
python main.py --mode train_predict --model-size medium
```

說明：
- 預設 `nano`，較適合 4GB 顯存（例如 T400）
- `medium` 在低顯存 GPU 上較容易 OOM

### 3.3 只推論（不重新訓練）

```bash
python main.py --mode predict_only
```

說明：
- 自動嘗試使用 `rfdetr_train_output` 中最新權重
- 若找不到自訓權重，改用官方預訓練權重

### 3.4 指定權重推論

```bash
python main.py --mode predict_only --predict-weights "F:/detr/sdsaliency900/rfdetr_train_output/your_model.pt"
```

### 3.5 設定推論閾值

```bash
python main.py --mode predict_only --threshold 0.5
```

### 3.6 訓練輪數（預設 50）

```bash
python main.py --mode train_predict --epochs 30
```

### 3.7 4GB 顯存建議指令

```bash
python main.py --mode train_predict --model-size nano --epochs 50
```

---

## 4. 輸入資料格式

訓練資料採 YOLO segmentation 標註：

- 影像：`images/train/*.png`、`images/val/*.png`
- 標註：`labels/train/*.txt`、`labels/val/*.txt`
- 每列格式：`class x1 y1 x2 y2 ...`（座標為 0~1）

程式會自動：
- 轉成 COCO 格式 `_annotations.coco.json`
- 將驗證集整理為 RF-DETR 預期的 `valid` split
- 將訓練影像 resize 至可被模型需求整除的尺寸

---

## 5. 常見問題排除

### 問題 1：`No module named 'pytorch_lightning'`

```bash
pip install pytorch-lightning
```

### 問題 2：`No module named 'pycocotools'`

```bash
pip install pycocotools
```

### 問題 3：`requires that faster-coco-eval is installed`

```bash
pip install faster-coco-eval
```

### 問題 4：`TensorBoard logging disabled`

```bash
pip install tensorboard
```

### 問題 5：`CUDA error: out of memory`

- 使用 `--model-size nano`
- `main.py` 已固定 `batch_size=1`
- 關閉其他佔用 GPU 的程式再訓練

### 問題 6：推論時出現
`Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)`

- 這是訓練結束後權重停在 CPU、推論輸入在 GPU 的裝置不一致問題
- 目前 `main.py` 已在推論前自動把權重搬回正確裝置

### 問題 7：存推論圖時出現
`TypeError: a bytes-like object is required, not 'Image'`

- `supervision` 標註器可能回傳 PIL 或 numpy
- 目前 `main.py` 已支援兩種輸出型態

### 問題 8：權重下載中斷（`ChunkedEncodingError`）

- 重新執行 `python main.py`
- 主程式已包含重試機制，通常可自動恢復

---

## 6. 注意事項

- 首次執行會依 `--model-size` 下載對應預訓練權重，耗時取決於網路
- 訓練需 CUDA GPU 才有合理速度
- `predict_only` 模式可避免每次都重新訓練

---

## 7. 建議工作流

1. 先跑一次完整訓練：

```bash
python main.py --mode train_predict
```

2. 後續只做推論：

```bash
python main.py --mode predict_only
```

