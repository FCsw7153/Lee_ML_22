import urllib.request
import zipfile
import os

def download_and_unzip_stdlib(url, save_path, extract_path='.'):
    """
    只使用标准库从 URL 下载并解压文件。
    """
    print(f"开始从 {url} 下载文件...")
    
    try:
        # --- 1. 下载文件 ---
        # urllib.request.urlretrieve 会直接将文件下载到指定路径
        urllib.request.urlretrieve(url, save_path)
        print(f"文件成功下载到: {save_path}")

        # --- 2. 解压文件 ---
        print(f"开始解压文件: {save_path}")
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        print(f"文件成功解压到: {extract_path}")

    except Exception as e:
        print(f"发生错误: {e}")

# --- 主程序 ---
if __name__ == "__main__":
    dropbox_url = "https://www.dropbox.com/s/m9q6273jl3djall/food-11.zip"
    zip_file_name = "food-11.zip"
    
    download_and_unzip_stdlib(url=dropbox_url, save_path=zip_file_name, extract_path='.')