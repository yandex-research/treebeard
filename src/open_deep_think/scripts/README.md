# IMO Answer Bench - Baseline Solver

This project provides a baseline solver for IMO (International Mathematical Olympiad) problems using API-based language models.

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

Run the baseline solver with the following command:

```bash
python -m open_deep_think.baseline_solve --start 0 --end 1 --model moonshotai/kimi-k2-thinking --max_tokens 64000 --output_path ../data
```
## Output Format

For each task, three files are created:

1. `Task_{task_id}_reasoning.txt` - The reasoning process from the model
2. `Task_{task_id}_solution.txt` - The final solution
3. `Task_{task_id}_response.json` - Full API response in JSON format


Then, Run the evaluation script:

```bash
python -m open_deep_think.scripts.evaluate --solutions_dir ../data/baseline/kimi-k2-thinking --judge_model google/gemini-3-flash-preview  --max_tokens 4096 --output_path ../data
```
