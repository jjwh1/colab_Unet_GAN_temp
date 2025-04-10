import torch
from models import TransCNN

from ptflops import get_model_complexity_info
from thop import profile
from torchsummary import summary

print("Initializing model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransCNN().to(device)
model.eval()

input_size = (3, 256, 256)  # 배치 크기 제외

# -------------------------------------------------
# 1️⃣ ptflops
print("\n[ptflops] -----------------------------")
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(model, input_size, as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    print(f"FLOPs: {macs} | Params: {params}")

# -------------------------------------------------
# 2️⃣ torchsummary
print("\n[torchsummary] -----------------------------")
summary(model, input_size=input_size)

# -------------------------------------------------
# 3️⃣ thop
print("\n[thop] -----------------------------")
dummy_input = torch.randn(1, *input_size).to(device)
macs, params = profile(model, inputs=(dummy_input,), verbose=False)
print(f"FLOPs: {macs / 1e9:.4f} GFLOPs | Params: {params / 1e6:.4f} M")
