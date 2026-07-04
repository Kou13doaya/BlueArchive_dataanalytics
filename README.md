# Blue Archive Data Analytics (総力戦・大決戦データ分析ツール)

ブルーアーカイブ（ブルアカ）の総力戦および大決戦のスコア分布データをビジュアライズし、統計分析を行うためのツールです。
StreamlitによるインタラクティブなWeb画面（GUI）と、コマンドラインから直接グラフ画像を生成するCLIツールの両方を提供しています。

---

## 🔧 必要環境・セットアップ

本ツールの実行には Python 3.9 以上が必要です。

### 1. 依存ライブラリのインストール
必要なモジュールをインストールしてください。
```bash
pip install -r requirements.txt
```

---

## 🚀 使い方

### 1. Streamlit アプリケーションの起動 (GUI)
ブラウザ上でグラフのフィルタリングや分析をインタラクティブに行うことができます。
```bash
streamlit run app.py
```
起動後、自動的にブラウザが開きます（デフォルト: `http://localhost:8501`）。

### 2. コマンドラインツールによるグラフ生成 (CLI)
特定イベントのグラフを直接画像として出力したい場合に使用します。
```bash
# 総力戦 (total_assault_89) のグラフを生成して保存
python run_analysis.py --event total_assault_89

# 大決戦 (grand_assault_34) のグラフを中位ブロック(Mid)で生成して保存
python run_analysis.py --event grand_assault_34 --mode Mid
```

---

## 📁 フォルダ構成と主要ファイル

```text
├── app.py                      # Streamlit Webアプリのエントリーポイント
├── run_analysis.py             # CLIツール（コマンドライン実行用）
├── data_loader.py              # 共通データローダー（ローカルキャッシュ優先）
├── requirements.txt            # 依存ライブラリ一覧
│
├── common/                     # 両ドメインで共有されるコア定義・ロジック
│   ├── event_metadata.py       # EVENT_META、翻訳用辞書、ID正規化処理など
│   └── score_converter.py      # スコアからクリアタイムへの逆算ロジックなど
│
├── OCR/                        # OCR（画像・動画認識）に関連するモジュール
│   ├── ocr_engine.py           # テンプレートマッチングによる画像解析エンジン
│   ├── ocr_parser.py           # EasyOCRを用いた画像パーサー
│   ├── video_ocr_parser.py      # スクロール動画からデータを抽出するパーサー
│   ├── build_templates.py      # テンプレート画像作成スクリプト
│   ├── binarize_templates.py   # テンプレート二値化前処理スクリプト
│   ├── ocr_specification.md    # OCRシステムの設計・高速化仕様書
│   ├── templates/              # OCR認識用数字画像アセット
│   └── debug/                  # ロジック調整・デバッグ用スクリプト群
│
├── analytics/                  # 統計・グラフ描画に関連するモジュール
│   ├── total_assault.py        # 総力戦のグラフ描画ロジック
│   ├── grand_assault.py        # 大決戦のグラフ描画ロジック
│   └── utils.py                # 分析・集計用の補助関数 (make_total_assault_summaryなど)
│
├── rank_data/                  # 各イベントのデータ格納先（Parquet形式）
├── image/                      # アイコンやトロフィーなどの画像アセット
├── verify_speed_and_correctness.py # OCR精度・速度のベンチマーク検証テスト
└── run_and_time.py             # OCR実行時間計測スクリプト
```

---

## 📊 データソースとローカルデータについて

### キャッシュの活用 (推奨)
データは `rank_data/rank_data_{event_id}.parquet` としてローカルにキャッシュされます。
キャッシュが存在する場合、インターネット接続なしで高速にデータをロードできます。
※ 推奨されるデータ追加方法：新しいイベントのスコアデータをParquet形式で `rank_data/` に直接配置してください。

### 規約に準拠したデータ取得方法 (推奨アプローチ)
利用規約（ https://bluearchive.jp/terms ）の禁止事項（通信情報の傍受、リバースエンジニアリング、BOT/自動化プログラムの利用など）を厳格に遵守するため、以下の方式によるデータ収集を検討・推奨しています。

* **録画動画からのOCR（画像認識）解析方式**:
  1. ユーザー自身がOSの標準機能（Windows Game BarやOBS等）を使用し、ゲーム内のランキング画面を手動でスクロールする動画（MP4等）をキャプチャします。
  2. 以下のコマンドで、録画した動画ファイルを読み込ませ、OCR機能（文字認識）により順位とスコアを自動抽出してParquet形式のデータを作成します。
  ```bash
  python OCR/video_ocr_parser.py --video <動画パス> --event <イベントID> --interval 0.1 --outdir rank_data
  ```
  * ※ ゲームの通信を傍受したりプログラムを改造することなく、また自動スクロール（BOT/マクロ）を行わないため、最も利用規約上の安全性が高い方法です。


