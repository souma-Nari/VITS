import tensorflow_datasets as tfds
from tensorflow_datasets.core.download import DownloadConfig

# 自分が保存した zip または解凍フォルダの親フォルダ
# (downloads/manual/VCTK-Corpus-0.92 を含むように構成)
manual_dir = "/home/souma/workspace/VCTK_Corpus"

# ビルダーを作成
builder = tfds.builder("vctk")

# ダウンロードを完全に手動に切り替える
builder.download_and_prepare(
    download_config=DownloadConfig(
        manual_dir=manual_dir,
        download_mode=tfds.download.GenerateMode.FORCE_REDOWNLOAD  # 明示的に強制準備
    )
)

# データセット読み込み
ds = builder.as_dataset(split="train")
