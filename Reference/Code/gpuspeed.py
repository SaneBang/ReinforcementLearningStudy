import torch
import time

# GPU가 사용 가능한지 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU를 사용할 수 없습니다. CPU를 사용합니다.")

# 테스트용 텐서 크기
size = 10000

# 랜덤 텐서 생성
tensor_a = torch.rand(size, size, device=device)
tensor_b = torch.rand(size, size, device=device)

# GPU를 미리 예열 (첫 연산은 보통 느릴 수 있음)
_ = torch.mm(tensor_a, tensor_b)

# 연산 시간 측정
start_time = time.time()
result = torch.mm(tensor_a, tensor_b)  # 행렬 곱셈
torch.cuda.synchronize()  # GPU 연산 대기
end_time = time.time()

print(f"Matrix multiplication time on {device}: {end_time - start_time:.6f} seconds")


import matplotlib.pyplot as plt
x = 1
y = 1
plt.plot(x)
plt.plot(y)
plt.show()
