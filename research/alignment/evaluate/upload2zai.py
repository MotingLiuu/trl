from zai import ZhipuAiClient

client = ZhipuAiClient(api_key="11307aa7012a4c118623c8f1133ecd1b.nJP8yyQKsEJFuvhU")

fileobject = client.files.create(
    file=open("data/evaluation_requests_zai_format_dpo.jsonl", "rb"),
    purpose="batch"
)
print(fileobject)