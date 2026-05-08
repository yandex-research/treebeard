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

Run parallel shards split by concurrency:

# Baseline
```bash
python -m open_deep_think.scripts.parallel_solve \
  --script baseline \
  --start 0 --end 400 \
  --concurrency 20 \
  --model openai/gpt-oss-120b  \
  --temperature 0.6 \
  --top_p 0.95 \
  --output_path ../data
  --run_name full_run
```


# IMO25 script
```bash
python -m open_deep_think.scripts.parallel_solve \
  --script imo25 \
  --start 0 --end 400 \
  --concurrency 20 \
  --model openai/gpt-oss-120b  \
  --temperature 0.6 \
  --top_p 0.95 \
  --output_path ../data
  --run_name full_run
```

# Tournament script
```bash
python -m open_deep_think.scripts.parallel_solve \
  --script tournament_merge_improve \
  --num_solutions 8 \
  --start 0 --end 400 \
  --concurrency 20 \
  --model openai/gpt-oss-120b  \
  --temperature 0.6 \
  --top_p 0.95 \
  --output_path ../data \
  --run_name full_run
```


## Output Format

Both solvers emit `Task_{task_id}_solution.txt`, which is what `evaluate.py` reads.


Then evaluate:

```bash
python -m open_deep_think.scripts.evaluate \
  --solutions_dir <dir> \
```
