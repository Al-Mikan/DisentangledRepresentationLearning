import torch
import sys

path = "train_result/2026-01-12/run_005/ablation/adversarial/dann/adversarial_dann_best.pth"
if len(sys.argv) > 1:
    path = sys.argv[1]

print(f"Loading {path}...")
try:
    state = torch.load(path, map_location="cpu")
    print("Keys found in state_dict:")
    keys = list(state.keys())
    # 最初の10個とDiscriminator関連を表示
    for k in keys[:10]:
        print(f" - {k}")
    print("...")
    
    disc_keys = [k for k in keys if "discriminator" in k]
    if disc_keys:
        print(f"✅ Found {len(disc_keys)} discriminator keys.")
        print(f"Example: {disc_keys[0]}")
    else:
        print("❌ No discriminator keys found!")
        
except Exception as e:
    print(f"Error: {e}")
