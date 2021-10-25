import time
import torch
import os

def main():
    
    print("Sleeping for 10 seconds")
    time.sleep(10)
    print("Available GPUs:", torch.cuda.device_count())
    print("All cores:", os.cpu_count())
    print("Available cores:", len(os.sched_getaffinity(0)))
    
if __name__ == "__main__":
    
    main()