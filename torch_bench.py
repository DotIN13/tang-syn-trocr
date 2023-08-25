import torch  
from torchvision.models import efficientnet_b0, vit_l_16, densenet161, regnet_y_1_6gf
from pytorch_benchmark import benchmark

print("#################### EfficientNet ##########################")
model = efficientnet_b0().cuda()
sample = torch.randn(64, 3, 224, 224)  # (B, C, H, W)
results = benchmark(model, sample, num_runs=10)
print(results)

print("###################### DenseNet ############################")
model2 =  densenet161().cuda()
sample2 = torch.randn(64, 3, 224, 224)  # (B, C, H, W)
results2 = benchmark(model2, sample2, num_runs=10)
print(results2)

print("####################### RegNet ############################")
model3 =  regnet_y_1_6gf().cuda()
sample3 = torch.randn(64, 3, 224, 224)  # (B, C, H, W)
results3 = benchmark(model3, sample3, num_runs=10)
print(results3)