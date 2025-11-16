# tire-rimline-profiler

> JP/EN: Tire 3D Rim-Line Zero Profiling toolkit built with Open3D + Python.

## 概要 / Overview

iPhone + ObjectCapture などで取得した GLB メッシュから、12 時方向の帯域を抽出し、リムラインを基準 0 とした X–Z' プロファイルを求める最小実装です。CLI と Jupyter Lab ノートブックを同梱し、単発解析・バッチ処理・再現性評価をすぐに試せます。

This repo extracts a narrow 12-o'clock slice from GLB meshes and builds an axial (X) versus radial (Z') profile whose zero is defined by a manually picked rim line. It ships a composable Python package, a CLI (`tireprof`), and three notebooks for quick experiments.

## インストール / Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .  # enables the `tireprof` command
```

## クイックスタート / Quick start

```bash
# 単一 GLB / single GLB
# optional: generate the synthetic sample mesh first
python scripts/create_synthetic_tire.py

tireprof --glb sample_data/synthetic_tire.glb \
         --tape-width 0.02 \
         --voxel 0.0015 --ransac-thresh 0.003 \
         --outer-band 0.05 --nbins 200 \
         --out runs/demo --save-debug

# フォルダ内の GLB を一括処理 / batch
tireprof --batch data/glbs --tape-width 0.02 --out runs/batch
```

CLI は必要に応じて Open3D のピッカーを起動し、SHIFT+クリックで 3〜10 点のリムラインを選択します。`--save-rim-points` を使うと JSON に記録でき、`--rim-json` で再利用できます。

## 座標系 / Coordinate system

* Cylinder RANSAC (`pyransac3d`) で軸を推定 → +X に整列
* 12 時方向（最大 Z 側）を +Z に合わせ、周方向の弧長を Y と定義
* 出力は `X`（軸方向）–`Z'`（リムライン基準の半径）プロファイル
* Rim line baseline: `Z0(Y) = α + β Y`。手動で選択した点を最小二乗回帰します。

## 同梱ノートブック / Included notebooks

| Notebook | 内容 / What it covers |
| --- | --- |
| `01_quickstart.ipynb` | 単一 GLB のロード、12 時スライス、リム基準化、Z' プロット |
| `02_batch_profiles.ipynb` | 複数 GLB の一括処理、プロファイル重ね描き・平均・分散 |
| `03_repeatability.ipynb` | 同一タイヤ再撮像ペアの RMSE・相関・差分カーブ |

全ノートブックには合成円筒 + 疑似リムラインを生成するセルがあり、GLB がなくても動作確認が可能です。

## データ / Data

* `sample_data/synthetic_tire.glb`: Open3D で生成した簡易円筒メッシュ。リムラインの段差を再現しています。
* `scripts/create_synthetic_tire.py`: 上記 GLB を再生成するスクリプト。

## 既知の制限 / Known limitations

* リムライン検出は手動ピックのみ。テクスチャや自動推定は今後の課題です。
* RANSAC が失敗する場合は `--voxel` や `--ransac-thresh` を調整してください。
* テープ幅や外周帯は実タイヤに合わせて調整が必要です。

## ライセンス / License

MIT License. See `LICENSE`.
