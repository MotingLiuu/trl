import json
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

input_filename = "data/generated_responses_dpo.jsonl"
output_filename = "data/evaluation_requests_zai_format_dpo.jsonl"


with open(input_filename, "r") as f:
    prompts, responses = [], []
    for line in f:
        record = json.loads(line)
        prompt = record["prompt"][0]["content"]
        response = record["response"][0]["content"]
        prompts.append(prompt)
        responses.append(response)

with open(output_filename, "w") as f:
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        record = {
            "custom_id": f"request-{i+1}",
            "method": "POST",
            "url": "/v4/chat/completions",
            "body": {
                "model": "glm-4-flash",
                "messages": [
                    {
                        "role": "system",
                        "content": '''
You are an expert evaluator for text summarization. Your task is to analyze a Reddit post and its corresponding 'TL;DR' summary, then provide a quantitative score for the summary's quality.

The user will provide the full text of the post, ending with the `TL;DR:` label, followed immediately by the generated summary.

**Your Task:**
Evaluate the generated summary on a scale of 0 to 10 based on the following three criteria.

1.  **Faithfulness (Accuracy) (Weight: 40%):**
    * Does the summary contain any factual errors or contradictions to the post?
    * A summary with hallucinations or clear factual errors (e.g., "Boy X is not talking to him" when the post says Girl X stopped talking) must be heavily penalized.
    * 10 = Perfectly accurate, no errors.
    * 0 = Completely false or contradictory.

2.  **Saliency (Informativeness) (Weight: 40%):**
    * Does the summary capture the *core conflict*, *main question*, or *central problem* of the post?
    * A summary that misses the main point (e.g., summarizing the "Family lawyer" post without mentioning the "siblings") must be heavily penalized.
    * 10 = Captures the essential core issue perfectly.
    * 0 = Misses the main point entirely.

3.  **Brevity & Fluency (Weight: 20%):**
    * Is the summary concise, grammatically correct, and easy to read?
    * It must be a "Too Long; Didn't Read" summary, not a long paragraph.
    * 10 = Fluent and highly concise.
    * 0 = Long-winded, unreadable, or incoherent.

**Output Format:**
You MUST provide your response *only* as a valid JSON object, with no other text before or after it. The final score should be a float from 0.0 to 10.0, calculated based on the weighted criteria.

{
  "score": <Your final 0-10 score (float)>,
  "reasoning": "<A brief, professional explanation for the score, referencing the three criteria.>",
  "scores_breakdown": {
    "faithfulness": <Score for faithfulness (0-10)>,
    "saliency": <Score for saliency (0-10)>,
    "brevity_fluency": <Score for brevity/fluency (0-10)>
  }
}
                            '''
                    },
                    {
                        "role": "user",
                        "content": prompt + response
                    },
                ],
                "temperature": 0.1,
            }
        }
        json_line = json.dumps(record, ensure_ascii=False)
        f.write(json_line + "\n")
    logger.info(f"Evaluation requests saved to {output_filename}.")


