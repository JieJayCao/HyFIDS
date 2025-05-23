import threading
import requests
import time
import os
from pathlib import Path

def get_filename_from_url(url):
    return url.split('/')[-1]

def download_worker(url):
    filename = get_filename_from_url(url)
    download_dir = Path("downloads")
    download_dir.mkdir(exist_ok=True)
    
    while True:
        try:
            file_path = download_dir / f"{filename}_{threading.current_thread().name}"
            with requests.get(url, stream=True, timeout=10) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                block_size = 8192
                downloaded = 0
                
                with open(file_path, 'wb') as f:
                    for data in r.iter_content(chunk_size=block_size):
                        f.write(data)
                        downloaded += len(data)
                        progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                        print(f"\r[{threading.current_thread().name}] 下载进度: {progress:.1f}%", end="")
                
                print(f"\n[{threading.current_thread().name}] 文件已保存到: {file_path}")
            break  # 下载完成后退出循环
        except Exception as e:
            print(f"[{threading.current_thread().name}] 错误: {e}")
            time.sleep(2)  # 出错稍等一下避免疯狂重连

def start_download_threads(url, num_threads=4):
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=download_worker, args=(url,), daemon=True, name=f"Downloader-{i+1}")
        t.start()
        threads.append(t)

    print(f"🚀 Started {num_threads} download threads targeting {url}")
    return threads

if __name__ == "__main__":
    # 指定下载链接，最好是大文件或 CDN 流
    test_url = "https://nbg1-speed.hetzner.com/1GB.bin"  # Hetzner测速文件
    num_threads = 4

    start_download_threads(test_url, num_threads)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Stopped by user.")