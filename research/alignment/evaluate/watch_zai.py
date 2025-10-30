import time
from zai import ZhipuAiClient

client = ZhipuAiClient(api_key="11307aa7012a4c118623c8f1133ecd1b.nJP8yyQKsEJFuvhU")

while True:
    batch_status = client.batches.retrieve("batch_1983977797710000128")
    print(f"任务状态: {batch_status.status}")
    
    if batch_status.status == "completed":
        print("任务完成！")
    elif batch_status.status in ["failed", "expired", "cancelled"]:
        print(f"任务失败，状态: {batch_status.status}")
        break
    
    time.sleep(30)  # 等待30秒后再次检查
    
    # 下载结果文件
    if batch_status.status == "completed":
        result_content = client.files.content(batch_status.output_file_id)
        result_content.write_to_file("data/batch_results_dpo.jsonl")
        print("结果文件下载完成: data/batch_results_dpo.jsonl")
        
        # 如果有错误文件，也可以下载
        if batch_status.error_file_id:
            error_content = client.files.content(batch_status.error_file_id)
            error_content.write_to_file("data/batch_errors_dpo.jsonl")
            print("错误文件下载完成: data/batch_errors_dpo.jsonl")