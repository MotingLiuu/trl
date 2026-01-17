import argparse
import json
import os
from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TLDR responses using vLLM with pairwise comparison")
    parser.add_argument("--model_name_or_path", type=str, default="casperhansen/llama-3.3-70b-instruct-awq", help="Path to the evaluation model (e.g., Llama-3-70B)")
    parser.add_argument("--baseline_file", type=str, default="data/gen_responses/generated_responses_0.6B_dpo6.jsonl", help="Path to the baseline JSONL file (Summary A)")
    parser.add_argument("--candidate_file", type=str, default="data/gen_responses/generated_responses_0.6B_dpo6_top5000.jsonl", help="Path to the candidate JSONL file (Summary B)")
    parser.add_argument("--output_file", type=str, default="data/gen_responses/pairwise_eval_results.jsonl", help="Path to save the evaluation results")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Fraction of GPU memory to use for the model executor")
    parser.add_argument("--max_model_len", type=int, default=1024, help="Maximum context length for the model")
    parser.add_argument("--max_samples", type=int, default=400, help="Limit evaluation to the first n samples")
    return parser.parse_args()

def create_pairwise_prompt(post, summary_a, summary_b):
    """
    Constructs the pairwise evaluation prompt using the user-provided template.
    """
    prompt_template = f"""**TL;DR win rate prompt**: Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? A good summary is both precise and concise. Post: {post} Summary A: {summary_a} Summary B: {summary_b} FIRST provide a one-sentence comparison of the two summaries, explaining which you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format: Comparison: <one-sentence comparison and explanation> Preferred: <"A" or "B">"""
    return prompt_template

def extract_content(item):
    """
    Helper to extract prompt and response content from the JSON object.
    Supports both list and string formats.
    """
    prompt = item.get('prompt', "")
    response = item.get('response', "")

    if isinstance(prompt, list):
        if len(prompt) > 0:
            prompt_content = prompt[0].get('content', "")
        else:
             prompt_content = ""
    else:
        prompt_content = prompt

    if isinstance(response, list):
        if len(response) > 0:
            response_content = response[0].get('content', "")
        else:
            response_content = ""
    else:
        response_content = response
    
    return prompt_content, response_content

def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # 1. Load Data
    logger.info(f"Loading baseline data from {args.baseline_file}...")
    baseline_data = []
    with open(args.baseline_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                baseline_data.append(json.loads(line))
    
    logger.info(f"Loading candidate data from {args.candidate_file}...")
    candidate_data = []
    with open(args.candidate_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                candidate_data.append(json.loads(line))

    if len(baseline_data) != len(candidate_data):
        logger.warning(f"Mismatch in data length: Baseline has {len(baseline_data)}, Candidate has {len(candidate_data)}. Will evaluate intersection based on index.")
    
    min_len = min(len(baseline_data), len(candidate_data))

    if args.max_samples is not None and args.max_samples > 0:
        min_len = min(min_len, args.max_samples)
        logger.info(f"Limiting evaluation to first {min_len} samples.")
    
    # 2. Prepare Prompts
    logger.info("Preparing evaluation prompts...")
    eval_prompts = []
    # Store indices to map back later
    eval_indices = []

    for i in range(min_len):
        base_item = baseline_data[i]
        cand_item = candidate_data[i]

        base_prompt, base_response = extract_content(base_item)
        cand_prompt, cand_response = extract_content(cand_item)

        # Sanity check: Prompts should match
        # Normalize slightly for comparison (strip whitespace) but proceed even if slightly different
        if base_prompt.strip() != cand_prompt.strip():
            # If prompts differ significantly, warn but proceed using baseline prompt as reference "Post"
            # In a real scenario, we might want to skip or ensure alignment by ID. 
            # For now, simplistic alignment by line number.
             pass

        # Clean "TL;DR:" from the Post text if present, to avoid confusion in the prompt
        post_text = base_prompt
        if "TL;DR:" in post_text:
             post_text = post_text.split("TL;DR:")[0].strip()

        # Construct Prompt
        # Summary A = Baseline, Summary B = Candidate
        full_prompt = create_pairwise_prompt(post_text, base_response, cand_response)
        eval_prompts.append(full_prompt)
        eval_indices.append(i)

    if not eval_prompts:
        logger.error("No valid prompts created.")
        return

    # 3. Load Model with vLLM
    logger.info(f"Loading model {args.model_name_or_path}...")
    llm = LLM(
        model=args.model_name_or_path, 
        tensor_parallel_size=args.tensor_parallel_size, 
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256) # Low temp for deterministic eval

    # 4. Generate Evaluations
    logger.info("Running inference...")
    outputs = llm.generate(eval_prompts, sampling_params)

    # 5. Process and Save Results
    logger.info(f"Saving results to {args.output_file}...")
    
    results_to_save = []

    for i, output in enumerate(outputs):
        original_idx = eval_indices[i]
        generated_text = output.outputs[0].text
        
        # Simple parser for "Preferred: A" or "Preferred: B"
        preferred = None
        if "Preferred: A" in generated_text:
            preferred = "A"
        elif "Preferred: B" in generated_text:
            preferred = "B"
        elif "Preferred: A" in generated_text.replace('"', ''): # Handle quotes
            preferred = "A"
        elif "Preferred: B" in generated_text.replace('"', ''):
            preferred = "B"
        
        result_item = {
            "index": original_idx,
            "post": eval_prompts[i].split("Post: ")[1].split(" Summary A:")[0], # Rough extraction for context or just use original
            "summary_a": baseline_data[original_idx].get("response", ""), # Store raw response object or just text
            "summary_b": candidate_data[original_idx].get("response", ""),
            "evaluation_prompt": eval_prompts[i],
            "evaluation_result": generated_text,
            "preferred": preferred,
            "file_a": args.baseline_file,
            "file_b": args.candidate_file
        }
        results_to_save.append(result_item)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in results_to_save:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Print quick stats
    a_wins = sum(1 for r in results_to_save if r['preferred'] == 'A')
    b_wins = sum(1 for r in results_to_save if r['preferred'] == 'B')
    ties_or_errors = len(results_to_save) - a_wins - b_wins
    
    logger.info(f"Evaluation Complete. Results saved to {args.output_file}")
    logger.info(f"Summary A (Baseline) Wins: {a_wins}")
    logger.info(f"Summary B (Candidate) Wins: {b_wins}")
    logger.info(f"Unclear/Ties: {ties_or_errors}")

if __name__ == "__main__":
    main()
