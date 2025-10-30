from zai import ZhipuAiClient

client = ZhipuAiClient(api_key="11307aa7012a4c118623c8f1133ecd1b.nJP8yyQKsEJFuvhU")

batch = client.batches.create(
    input_file_id="1761852035_c095e949a23a48e09182c239dae5c1f9",
    endpoint="/v4/chat/completions",
    auto_delete_input_file=False,
    metadata={"project": "alignment_evaluation", "type": "dpo_summaries"}
)
print(batch)
