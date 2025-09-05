import requests
import threading
import time

def send_request(request_id):
    url = "http://0.0.0.0:11996/tts_fast"
    data = {
        "text": f"Mason is a talented badminton player, and we're focusing on enhancing his smash technique. To truly unlock his potential, we'll emphasize full body rotation from his legs and hips for explosive power, refine his wrist snap for maximum shuttlecock speed, and ensure a strong, forward follow-through for accuracy. At home, he can practice shadow swings, concentrating on a fluid, coordinated motion from core to racket. This will greatly enhance his power and court control.",
        "audio_paths": ["tests/sample_prompt.wav"],
        "max_text_tokens_per_sentence": 120,
        "sentences_bucket_max_size": 10
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=data)
        if response.status_code == 200:
            filename = f"output_{request_id}.wav"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"请求 {request_id} 完成, 耗时 {time.time() - start_time} 秒")
        else:
            print(f"请求 {request_id} 失败")
    except Exception as e:
        print(f"请求 {request_id} 异常: {e}")

# 使用信号量控制并发数
semaphore = threading.Semaphore(1)

def worker(request_id):
    with semaphore:
        send_request(request_id)

# 创建并启动8个线程
threads = []
for i in range(1):
    thread = threading.Thread(target=worker, args=(i,))
    threads.append(thread)
    thread.start()

# 等待所有线程完成
for thread in threads:
    thread.join()

print("所有请求完成")