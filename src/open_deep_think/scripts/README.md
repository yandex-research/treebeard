# IMO AnswerBench Scripts

This folder contains CLI scripts for generating and evaluating IMO AnswerBench solutions.

## Configuration

Set up your OpenRouter API credentials as environment variables:

```bash
export API_KEY="your-api-key-here"
```

Optionally, you can set a custom API base URL:

```bash
export API_BASE_URL="https://your-custom-url.com/api/v1"
```

## Usage

Run the baseline solver:

```bash
python -m open_deep_think.scripts.baseline_solve \
  --start 0 --end 1 \
  --model moonshotai/kimi-k2-thinking \
  --max_tokens 64000 \
  --temperature 1.0 \
  --top_p 0.95 \
  --output_path data/
```

Run the IMO25 verification-and-refinement pipeline reproduction:

```bash
python -m open_deep_think.scripts.imo25_solve \
  --start 0 --end 1 \
  --model moonshotai/kimi-k2-thinking \
  --temperature 0.1 \
  --top_p 1.0 \
  --output_path logs/
```

Run parallel shards split by concurrency:

```bash
python -m open_deep_think.scripts.parallel_imo25_solve \
  --script imo25 \
  --start 0 --end 10 \
  --concurrency 3 \
  --model moonshotai/kimi-k2-thinking \
  --temperature 0.1 \
  --top_p 1.0 \
  --output_path logs/
```

Run baseline in parallel with the same launcher:

```bash
python -m open_deep_think.scripts.parallel_imo25_solve \
  --script baseline \
  --start 0 --end 10 \
  --concurrency 3 \
  --model moonshotai/kimi-k2-thinking \
  --baseline_max_tokens 64000 \
  --temperature 1.0 \
  --top_p 0.95 \
  --output_path logs/
```

## Output Format

Both solvers emit `Task_{task_id}_solution.txt`, which is what `evaluate.py` reads.

The IMO25 script also emits:

1. `Task_{task_id}_reasoning.txt` - extracted `<think>...</think>` content from final solver output.
2. `Task_{task_id}_response.json` - full final solver API response.
3. `Task_{task_id}_progress.json` - run/iteration timeline for that task.
4. `Task_{task_id}_llm_outputs.jsonl` - every LLM call request/response for that task.
5. `all_llm_outputs.jsonl` - every LLM call across all tasks in the run.

Then evaluate:

```bash
python -m open_deep_think.scripts.evaluate \
  --solutions_dir logs/imo25/kimi-k2-thinking/<run_name> \
  --judge_model models/gemini-3-flash-preview \
  --max_tokens 4096
```
