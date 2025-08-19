import torch
import utils
from models import SynthesizerTrn
from text.symbols import symbols

# --- ▼▼▼ あなたの環境に合わせて、この2つのパスを修正してください ▼▼▼ ---
CONFIG_PATH = "/home/souma/workspace/configs/vctk_base.json"  # 学習に使っている設定ファイル
PRETRAINED_PATH = "/home/souma/workspace/pretrained_files/vits/pretrained_vctk.pth" # 事前学習済みモデルのパス
# --- ▲▲▲ パスの修正はここまで ▲▲▲ ---

def main():
    print("--- Key Mismatch Debugger ---")
    
    # 1. ローカルのモデル定義をロード
    print(f"\n[1] Loading local model definition from {CONFIG_PATH}...")
    hps = utils.get_hparams_from_file(CONFIG_PATH)
    local_net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    local_net_g.eval()
    model_keys = set(local_net_g.state_dict().keys())
    print(f"Local model has {len(model_keys)} keys.")

    # 2. 事前学習済みモデルの state_dict をロード
    print(f"\n[2] Loading pretrained model from {PRETRAINED_PATH}...")
    checkpoint_dict = torch.load(PRETRAINED_PATH, map_location="cpu")
    
    # "model" キーの有無をチェックして state_dict を取得
    pretrained_state_dict = checkpoint_dict.get("model", checkpoint_dict)
    pretrained_keys = set(pretrained_state_dict.keys())
    print(f"Pretrained model has {len(pretrained_keys)} keys.")

    # 3. キーを比較
    print("\n[3] Comparing keys...")
    
    # 一致するキー
    matching_keys = model_keys.intersection(pretrained_keys)
    
    # ローカルモデルにしか存在しないキー
    local_only_keys = model_keys - pretrained_keys
    
    # 事前学習済みモデルにしか存在しないキー
    pretrained_only_keys = pretrained_keys - model_keys

    print("\n--- [ ANALYSIS RESULT ] ---")
    print(f"Matching Keys: {len(matching_keys)}")
    print(f"Keys ONLY in Local Model: {len(local_only_keys)}")
    print(f"Keys ONLY in Pretrained Model: {len(pretrained_only_keys)}")
    print("---------------------------\n")

    if not matching_keys:
        print("!!! CRITICAL: No matching keys found. This is why loading is failing.")
        print("This often happens if one model has a 'module.' prefix and the other does not.\n")

        print("--- Keys in Local Model (sample) ---")
        for i, key in enumerate(list(local_keys)[:5]):
            print(key)

        print("\n--- Keys in Pretrained Model (sample) ---")
        for i, key in enumerate(list(pretrained_keys)[:5]):
            print(key)
    else:
        print("Success: Found matching keys. Loading should be possible.")


if __name__ == "__main__":
    main()