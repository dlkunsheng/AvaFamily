import psutil
import torch

def check_system_resources():
    # 检查内存
    memory = psutil.virtual_memory()
    print(f"Total Memory: {memory.total / (1024 ** 3):.2f} GB")
    print(f"Available Memory: {memory.available / (1024 ** 3):.2f} GB")

    # 检查CPU
    cpu_count = psutil.cpu_count(logical=True)
    print(f"CPU Cores: {cpu_count}")

    # 检查GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB")
    else:
        print("No GPU available.")

    # 检查存储空间
    disk = psutil.disk_usage('/')
    print(f"Total Disk Space: {disk.total / (1024 ** 3):.2f} GB")
    print(f"Free Disk Space: {disk.free / (1024 ** 3):.2f} GB")

if __name__ == "__main__":
    check_system_resources()