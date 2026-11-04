import torch

print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
print("cuda.is_available():", torch.cuda.is_available())
print("cuda.device_count(): ", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  sm_{p.major}{p.minor}  {p.total_memory / 2**30:.1f} GiB  {p.multi_processor_count} SMs")
