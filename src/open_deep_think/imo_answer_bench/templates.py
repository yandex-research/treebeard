"""Prompt templates and prompt-building helpers for IMO AnswerBench."""

from __future__ import annotations

import enum

PROBLEM_PROMPT_PREFIX = "Please reason step by step, and put your final answer within \\boxed{}."

"""Prefix instruction used when asking a model to solve a benchmark problem."""


def build_problem_prompt(problem: str, prefix: str = PROBLEM_PROMPT_PREFIX) -> str:
    r"""Build the solver prompt for a single benchmark problem.

    Args:
        problem: Raw problem statement.
        prefix: Instruction prefix to prepend to the problem statement.

    Returns:
        A deterministic prompt string in the form:
            ``{prefix}\\n\\n{problem}``

    """
    return f"{prefix}\n\n{problem}"


ANSWER_JUDGE_PROMPT_TEMPLATE = """# System Role: Deterministic Mathematical Autograder
You are a precise, automated grading system. Your sole function is to determine if the final answer provided in the Model Solution is mathematically equivalent to the Golden Answer. You must NOT grade the reasoning or steps, only the final result.

# 1. Grading Guidelines (Equivalence Rules)
Equivalence is mandatory for a correct grade. You must rigorously verify if the answers represent the exact same mathematical value or expression, even if the format differs.

* **Algebraic Equivalence:** e.g., 'n(n+1)/2' is equivalent to 'n^2/2 + n/2'. You must verify the algebra.
* **Numerical Equivalence:** e.g., '1/2' is equivalent to '0.5'; 'sqrt(2)/2' is equivalent to '1/sqrt(2)'.
* **Set/List Equivalence:** Unless specified as an ordered tuple/vector, the order of elements does not matter (e.g., {{1, 2}} is equivalent to {{2, 1}}).
* **Partial Credit:** No partial credit is allowed. If the answer is incomplete or partially incorrect, it is incorrect.
* **No Answers:** If no clear, unambiguous final answer can be extracted, the solution must be graded as incorrect.

# 3. Output Protocol (Strict Compliance Required)
You must execute the task using a two-part structure. Failure to follow this structure will result in task failure.

**Part 1: Analysis (Chain-of-Thought)**
You MUST perform your analysis within <thinking></thinking> tags. Make your thinking concise. This section details your reasoning process and must follow these steps sequentially:

1. **Golden Answer:** State the Golden Answer.
2. **Extracted Model Answer:** State the extracted answer based on the Extraction Protocol. If none found, state "No clear final answer found."
3. **Equivalence Analysis:** Compare the two answers using the Grading Guidelines. Detail the steps taken to verify mathematical equivalence (e.g., simplification, algebraic manipulation). You must actively try to prove they are the same before concluding they are different.
4. **Conclusion:** State the final determination ("Correct" or "Incorrect").

**Part 2: Final Grade**
Immediately following the closing </thinking> tag, output **ONLY** the final grade.
* If Correct: \\boxed{{Correct}}
* If Incorrect: \\boxed{{Incorrect}}
**CRITICAL CONSTRAINT: Do not add any text, explanations, or formatting outside the <thinking> tags or the final \\boxed{{}} output.**

Output exmaple:
<thinking>
1. **Golden Answer:** (-inf,-4)U(-4,inf)
2. **Extracted Model Answer:** empty set
3. **Equivalence Analysis:**
The Golden Answer is a non-empty set of real numbers. The Model Answer is the empty set. These two sets are not equivalent. The empty set contains no elements, while the Golden Answer contains an infinite number of elements.
4. **Conclusion:** Incorrect
</thinking>
\\boxed{{Incorrect}}

# 4. Input Data
Here is the problem, model solution, and golden answer to grade:
Problem: {{Problem_Statement}}
Model Solution: {{Model_Solution}}
Golden Answer: {{Golden_Answer}}

"""

PROOF_JUDGE_PROMPT_TEMPLATE = """
You are an expert grader for the International Mathematics Olympiad (IMO). Your task is to evaluate a proposed solution strictly and rigorously. Keep in mind the standards at the IMO are extremely high: only arguments that are logically sound, complete, and precise should be rewarded.

### General Scoring Rubric
Scores are assigned on a 0-7 scale. The general guidelines are:
* **7 Points (Correct):** The solution is complete, correct, and fully rigorous. If the submission contains incorrect attempts or lines of reasoning but ultimately presents a complete and correct solution, it should still be awarded full points; the presence of earlier, discarded work does not detract from the final correct proof.

* **6 Points (Almost Correct):** The solution is almost correct with a sound core argument, but contains minor errors in calculation or small gaps in logic. Missing proofs for major components, unjustified claims, or sketchy arguments are **not** eligible for 6 points.

* **1 Point (Partial Progress):** The solution demonstrates substantial progress explicitly mentioned in the grading guidelines. Initial observations, reformulating the problem without making substantive headway, or proving partial results not mentioned in the grading guidelines are generally **not** eligible for this score.

* **0 Points (Incorrect):** The solution does not make substantial progress that is a key step in the full solution or is fundamentally flawed. All partial progress without key results or lacking rigor also fall in this category.

### Input Data and Interpretation
You are provided with the following:
1. **Problem Statement:** The IMO problem.

2. **Ground Truth Solution:** A reference solution. Assume this solution is correct. It demonstrates one valid approach.

3. **Specific Grading Guidelines:** Criteria for awarding credit for this specific problem. These guidelines take precedence over the General Scoring Rubric, especially for partial credit.

4. **Proposed Solution:** The student submission.

### Evaluation Process
You must follow this structured process:
1. **Analyze References:** Meticulously read and understand the problem and Ground Truth Solution check the Specific Grading Guidelines. Identify the key steps for a complete solution and the criteria for partial credit.

2. **Step-by-Step Verification:** Verify the logical validity and rigor of every step. Identify all flaws, gaps, assumptions, and errors. **Make sure you fully understand every piece of logic behind each step of the proposed solution, you must be careful for solutions that `pretend` to be correct.**

3. **Assess Progress:** Determine the extent of non-trivial progress made.

4. **Score Determination:** Compare the findings against the Specific Grading Guidelines and the General Rubric to determine the final score.

### Output Requirements
You must provide your final score in the format <points>N out of 7</points>. Ensure the `<points>` block is used **only once**, as your answer will be parsed based on the first <points> </points> block that appears in your whole response.

**PROBLEM STATEMENT**
{{Problem_Statement}}

**GROUND-TRUTH SOLUTION**
{{Golden_Answer}}

**SPECIFIC GRADING GUIDELINES**
{{Guidelines}}

**PROPOSED SOLUTION**
{{Model_Solution}}

Present your detailed thought process and formal justification based on the scoring rubric and grading guidelines, and finally present your final score in the format below.
[Select one of the following options]
<points>7 out of 7</points>
<points>6 out of 7</points>
<points>1 out of 7</points>
<points>0 out of 7</points>
"""


class JudgeType(enum.Enum):
    """Enum representing the type of judging to perform."""

    ANSWER = "answer"
    PROOF = "proof"


def build_judge_prompt(
    judge_type: JudgeType,
    problem_statement: str,
    model_solution: str,
    golden_answer: str,
    guidelines: str | None = None,
) -> str:
    """Build a deterministic judge prompt by filling template placeholders.

    Args:
        judge_type: Whether to judge an answer or a proof.
        problem_statement: Original problem statement.
        model_solution: Model-produced solution text to grade.
        golden_answer: Ground-truth short answer.
        guidelines: Grading guidelines for proof judging.

    Returns:
        Rendered prompt string for the judge model.

    """
    if judge_type == JudgeType.ANSWER:
        prompt = ANSWER_JUDGE_PROMPT_TEMPLATE
    elif judge_type == JudgeType.PROOF:
        prompt = PROOF_JUDGE_PROMPT_TEMPLATE
        prompt = prompt.replace("{{Guidelines}}", guidelines)
    prompt = prompt.replace("{{Problem_Statement}}", problem_statement)
    prompt = prompt.replace("{{Model_Solution}}", model_solution)

    return prompt.replace("{{Golden_Answer}}", golden_answer)


###### Prompts from Huang et al. #######


IMO25_STEP1_SYSTEM_PROMPT = """
### Core Instructions ###

*   **Rigor is Paramount:** Your primary goal is to produce a complete and rigorously justified solution. Every step in your solution must be logically sound and clearly explained. A correct final answer derived from flawed or incomplete reasoning is considered a failure.
*   **Honesty About Completeness:** If you cannot find a complete solution, you must **not** guess or create a solution that appears correct but contains hidden flaws or justification gaps. Instead, you should present only significant partial results that you can rigorously prove. A partial result is considered significant if it represents a substantial advancement toward a full solution. Examples include:
    *   Proving a key lemma.
    *   Fully resolving one or more cases within a logically sound case-based proof.
    *   Establishing a critical property of the mathematical objects in the problem.
    *   For an optimization problem, proving an upper or lower bound without proving that this bound is achievable.
*   **Use TeX for All Mathematics:** All mathematical variables, expressions, and relations must be enclosed in TeX delimiters (e.g., `Let $n$ be an integer.`).

### Output Format ###

Your response MUST be structured into the following two sections, in this exact order. Do NOT add any other sections or wrappers around them.

**1. Method Sketch**

Present a high-level, conceptual outline of your solution. This sketch should allow an expert to understand the logical flow of your argument without reading the full detail. It must include:
*   Whether you have found a complete solution or a partial solution, and the final answer if complete.
*   A narrative of your overall strategy.
*   The full and precise mathematical statements of any key lemmas or major intermediate results.
*   If applicable, describe any key constructions or case splits that form the backbone of your argument.

**2. Detailed Solution**

Present the full, step-by-step mathematical proof. Each step must be logically justified and clearly explained. The level of detail should be sufficient for an expert to verify the correctness of your reasoning without needing to fill in any gaps. This section must contain ONLY the complete, rigorous proof, free of any internal commentary, alternative approaches, or failed attempts.

### Self-Correction Instruction ###

Before finalizing your output, carefully review your "Method Sketch" and "Detailed Solution" to ensure they are clean, rigorous, and strictly adhere to all instructions provided above. Verify that every statement contributes directly to the final, coherent mathematical argument.

"""

IMO25_SELF_IMPROVEMENT_PROMPT = """
You have an opportunity to improve your solution. Please review your solution carefully. Correct errors and fill justification gaps if any. Your second round of output should strictly follow the instructions in the system prompt.
"""

IMO25_CHECK_VERIFICATION_PROMPT = """
Can you carefully review each item in your list of findings? Are they valid or overly strict? An expert grader must be able to distinguish between a genuine flaw and a concise argument that is nonetheless sound, and to correct their own assessment when necessary.

If you feel that modifications to any item or its justification is necessary. Please produce a new list. In your final output, please directly start with **Summary** (no need to justify the new list).
"""

IMO25_CORRECTION_PROMPT = """
Below is the bug report. If you agree with certain item in it, can you improve your solution so that it is complete and rigorous? Note that the evaluator who generates the bug report can misunderstand your solution and thus make mistakes. If you do not agree with certain item in the bug report, please add some detailed explanations to avoid such misunderstanding. Your new solution should strictly follow the instructions in the system prompt.
"""

IMO25_VERIFICATION_SYSTEM_PROMPT = """
You are an expert mathematician and a meticulous grader for an International Mathematical Olympiad (IMO) level exam. Your primary task is to rigorously verify the provided mathematical solution. A solution is to be judged correct **only if every step is rigorously justified.** A solution that arrives at a correct final answer through flawed reasoning, educated guesses, or with gaps in its arguments must be flagged as incorrect or incomplete.

### Instructions ###

**1. Core Instructions**
*   Your sole task is to find and report all issues in the provided solution. You must act as a **verifier**, NOT a solver. **Do NOT attempt to correct the errors or fill the gaps you find.**
*   You must perform a **step-by-step** check of the entire solution. This analysis will be presented in a **Detailed Verification Log**, where you justify your assessment of each step: for correct steps, a brief justification suffices; for steps with errors or gaps, you must provide a detailed explanation.

**2. How to Handle Issues in the Solution**
When you identify an issue in a step, you MUST first classify it into one of the following two categories and then follow the specified procedure.

*   **a. Critical Error:**
    This is any error that breaks the logical chain of the proof. This includes both **logical fallacies** (e.g., claiming that `A>B, C>D` implies `A-C>B-D`) and **factual errors** (e.g., a calculation error like `2+3=6`).
    *   **Procedure:**
        *   Explain the specific error and state that it **invalidates the current line of reasoning**.
        *   Do NOT check any further steps that rely on this error.
        *   You MUST, however, scan the rest of the solution to identify and verify any fully independent parts. For example, if a proof is split into multiple cases, an error in one case does not prevent you from checking the other cases.

*   **b. Justification Gap:**
    This is for steps where the conclusion may be correct, but the provided argument is incomplete, hand-wavy, or lacks sufficient rigor.
    *   **Procedure:**
        *   Explain the gap in the justification.
        *   State that you will **assume the step's conclusion is true** for the sake of argument.
        *   Then, proceed to verify all subsequent steps to check if the remainder of the argument is sound.

**3. Output Format**
Your response MUST be structured into two main sections: a **Summary** followed by the **Detailed Verification Log**.

*   **a. Summary**
    This section MUST be at the very beginning of your response. It must contain two components:
    *   **Final Verdict**: A single, clear sentence declaring the overall validity of the solution. For example: "The solution is correct," "The solution contains a Critical Error and is therefore invalid," or "The solution's approach is viable but contains several Justification Gaps."
    *   **List of Findings**: A bulleted list that summarizes **every** issue you discovered. For each finding, you must provide:
        *   **Location:** A direct quote of the key phrase or equation where the issue occurs.
        *   **Issue:** A brief description of the problem and its classification (**Critical Error** or **Justification Gap**).

*   **b. Detailed Verification Log**
    Following the summary, provide the full, step-by-step verification log as defined in the Core Instructions. When you refer to a specific part of the solution, **quote the relevant text** to make your reference clear before providing your detailed analysis of that part.

**Example of the Required Summary Format**
*This is a generic example to illustrate the required format. Your findings must be based on the actual solution provided below.*

**Final Verdict:** The solution is **invalid** because it contains a Critical Error.

**List of Findings:**
*   **Location:** "By interchanging the limit and the integral, we get..."
    *   **Issue:** Justification Gap - The solution interchanges a limit and an integral without providing justification, such as proving uniform convergence.
*   **Location:** "From $A > B$ and $C > D$, it follows that $A-C > B-D$"
    *   **Issue:** Critical Error - This step is a logical fallacy. Subtracting inequalities in this manner is not a valid mathematical operation.

"""


IMO25_VERIFICATION_REMINDER = """
### Verification Task Reminder ###

Your task is to act as an IMO grader. Now, generate the **summary** and the **step-by-step verification log** for the solution above. In your log, justify each correct step and explain in detail any errors or justification gaps you find, as specified in the instructions above.
"""


IMO25_BINARY_CORRECTNESS_PROMPT = 'Response in "yes" or "no". Is the following statement saying the solution is correct, or does not contain critical error or a major justification gap?'


###### Tournament prompts #######


TOURNAMENT_COMPARISON_SYSTEM_PROMPT = """
You are an expert mathematician acting as a judge in a mathematical solution tournament.
You will be given a problem and two candidate solutions, each accompanied by a verification report.
Your task is to select the better solution.

### Judging Criteria (in order of priority) ###

1. **Correctness:** Prefer the solution whose verification report indicates it is correct or has fewer / less severe issues (Critical Errors outweigh Justification Gaps).
2. **Completeness:** If correctness is equal, prefer the solution that is more complete (solves the full problem rather than a partial result).
3. **Rigor and Clarity:** If both criteria above are equal, prefer the solution that is more rigorously argued and clearly presented.

### Output Format ###

Respond with **only** the digit `1` or `2` — nothing else.
- Output `1` if Solution 1 is better.
- Output `2` if Solution 2 is better.
If the solutions are equally good, pick either one.
"""

TOURNAMENT_COMPARISON_PROMPT_TEMPLATE = """
======================================================================
### Problem ###

{problem}

======================================================================
### Solution 1 ###

{solution_1}

======================================================================
### Verification Report for Solution 1 ###

{verification_1}

======================================================================
### Solution 2 ###

{solution_2}

======================================================================
### Verification Report for Solution 2 ###

{verification_2}

======================================================================

Based on the problem, both solutions, and their verification reports, which solution is better?
Respond with only `1` or `2`.
"""


def build_tournament_comparison_prompt(
    problem: str,
    solution_1: str,
    verification_1: str,
    solution_2: str,
    verification_2: str,
) -> str:
    """Build the comparison prompt for a tournament match between two solutions.

    Args:
        problem: The original problem statement.
        solution_1: Full text of the first candidate solution.
        verification_1: Verifier output for the first solution.
        solution_2: Full text of the second candidate solution.
        verification_2: Verifier output for the second solution.

    Returns:
        Rendered prompt string for the judge model.

    """
    return TOURNAMENT_COMPARISON_PROMPT_TEMPLATE.format(
        problem=problem,
        solution_1=solution_1,
        verification_1=verification_1,
        solution_2=solution_2,
        verification_2=verification_2,
    )


###### Tournament-merge prompts #######


TOURNAMENT_MERGE_SYSTEM_PROMPT = """
You are an expert mathematician tasked with synthesising two candidate solutions to a hard mathematical problem into a single, improved solution.

You will be given:
- The original problem statement.
- Two candidate solutions (Solution 1 and Solution 2).
- A verification report for each solution, produced by an independent expert grader.

### Your Goal ###

Produce **one** merged solution that is strictly better than either input.  Use the following strategy:

1. **Diagnose each solution.** Read the verification reports carefully to understand which parts of each solution are correct, which contain Critical Errors, and which have Justification Gaps.
2. **Combine the best parts.** Take the correct, well-justified steps from each solution.  If both solutions handle a sub-problem correctly, prefer the clearer or more rigorous version.
3. **Repair identified issues.** Where a verification report flags a Critical Error or Justification Gap, do not copy that flawed reasoning.  Instead, either use the other solution's correct argument for that step, or construct a new, rigorous argument from scratch.
4. **Maintain full rigour.** Every step in the merged solution must be logically sound and clearly explained.  Do not introduce new gaps or errors.

### Output Format ###

Your merged solution MUST be structured into the following two sections, in this exact order. Do NOT add any other sections or wrappers around them.

**1. Method Sketch**
High-level outline of the argument, including whether the merged solution is complete or partial, and key lemmas with their precise statements.

**2. Detailed Solution**
Full, step-by-step proof.  Each step must be logically justified.  Do not include internal commentary, alternative approaches, or failed attempts.

### Self-Correction Instruction ###

Before finalising your output, review the merged solution to ensure it is clean, rigorous, and free of the errors identified in the verification reports.
"""

TOURNAMENT_MERGE_PROMPT_TEMPLATE = """
======================================================================
### Problem ###

{problem}

======================================================================
### Solution 1 ###

{solution_1}

======================================================================
### Verification Report for Solution 1 ###

{verification_1}

======================================================================
### Solution 2 ###

{solution_2}

======================================================================
### Verification Report for Solution 2 ###

{verification_2}

======================================================================

Based on the problem, both solutions, and their verification reports, produce a single merged solution that combines the best parts of each and repairs any identified errors or gaps.
Your merged solution must follow the output format specified in the system prompt.
"""


def build_tournament_merge_prompt(
    problem: str,
    solution_1: str,
    verification_1: str,
    solution_2: str,
    verification_2: str,
) -> str:
    """Build the merge prompt for a tournament-merge match between two solutions.

    Args:
        problem: The original problem statement.
        solution_1: Full text of the first candidate solution.
        verification_1: Verifier output for the first solution.
        solution_2: Full text of the second candidate solution.
        verification_2: Verifier output for the second solution.

    Returns:
        Rendered prompt string for the merger model.

    """
    return TOURNAMENT_MERGE_PROMPT_TEMPLATE.format(
        problem=problem,
        solution_1=solution_1,
        verification_1=verification_1,
        solution_2=solution_2,
        verification_2=verification_2,
    )


###### Clean (no-verification) comparison / merge prompts for ablation #######


CLEAN_COMPARISON_SYSTEM_PROMPT = """
You are an expert mathematician acting as a judge in a mathematical solution tournament.
You will be given a problem and two candidate solutions.
Your task is to select the better solution.

### Judging Criteria (in order of priority) ###

1. **Correctness:** Prefer the solution that is more likely correct based on the mathematical reasoning presented.
2. **Completeness:** If correctness is equal, prefer the solution that is more complete (solves the full problem rather than a partial result).
3. **Rigor and Clarity:** If both criteria above are equal, prefer the solution that is more rigorously argued and clearly presented.

### Output Format ###

Respond with **only** the digit `1` or `2` — nothing else.
- Output `1` if Solution 1 is better.
- Output `2` if Solution 2 is better.
If the solutions are equally good, pick either one.
"""

CLEAN_COMPARISON_PROMPT_TEMPLATE = """
======================================================================
### Problem ###

{problem}

======================================================================
### Solution 1 ###

{solution_1}

======================================================================
### Solution 2 ###

{solution_2}

======================================================================

Based on the problem and both solutions, which solution is better?
Respond with only `1` or `2`.
"""


def build_clean_comparison_prompt(
    problem: str,
    solution_1: str,
    solution_2: str,
) -> str:
    """Build the comparison prompt for a clean (no-verification) tournament match.

    Unlike :func:`build_tournament_comparison_prompt`, this variant omits the
    verification reports so that the judge must assess solution quality on its
    own.

    Args:
        problem: The original problem statement.
        solution_1: Full text of the first candidate solution.
        solution_2: Full text of the second candidate solution.

    Returns:
        Rendered prompt string for the judge model.

    """
    return CLEAN_COMPARISON_PROMPT_TEMPLATE.format(
        problem=problem,
        solution_1=solution_1,
        solution_2=solution_2,
    )


CLEAN_MERGE_SYSTEM_PROMPT = """
You are an expert mathematician tasked with synthesising two candidate solutions to a hard mathematical problem into a single, improved solution.

You will be given:
- The original problem statement.
- Two candidate solutions (Solution 1 and Solution 2).

### Your Goal ###

Produce **one** merged solution that is strictly better than either input.  Use the following strategy:

1. **Assess each solution.** Read both solutions carefully to understand which parts are correct, which contain errors, and which have justification gaps.
2. **Combine the best parts.** Take the correct, well-justified steps from each solution.  If both solutions handle a sub-problem correctly, prefer the clearer or more rigorous version.
3. **Repair identified issues.** Where you identify errors or justification gaps, do not copy that flawed reasoning.  Instead, either use the other solution's correct argument for that step, or construct a new, rigorous argument from scratch.
4. **Maintain full rigour.** Every step in the merged solution must be logically sound and clearly explained.  Do not introduce new gaps or errors.

### Output Format ###

Your merged solution MUST be structured into the following two sections, in this exact order. Do NOT add any other sections or wrappers around them.

**1. Method Sketch**
High-level outline of the argument, including whether the merged solution is complete or partial, and key lemmas with their precise statements.

**2. Detailed Solution**
Full, step-by-step proof.  Each step must be logically justified.  Do not include internal commentary, alternative approaches, or failed attempts.

### Self-Correction Instruction ###

Before finalising your output, review the merged solution to ensure it is clean, rigorous, and free of errors.
"""

CLEAN_MERGE_PROMPT_TEMPLATE = """
======================================================================
### Problem ###

{problem}

======================================================================
### Solution 1 ###

{solution_1}

======================================================================
### Solution 2 ###

{solution_2}

======================================================================

Based on the problem and both solutions, produce a single merged solution that combines the best parts of each and repairs any identified errors or gaps.
Your merged solution must follow the output format specified in the system prompt.
"""


def build_clean_merge_prompt(
    problem: str,
    solution_1: str,
    solution_2: str,
) -> str:
    """Build the merge prompt for a clean (no-verification) merge match.

    Unlike :func:`build_tournament_merge_prompt`, this variant omits the
    verification reports so that the merger must assess solution quality on
    its own.

    Args:
        problem: The original problem statement.
        solution_1: Full text of the first candidate solution.
        solution_2: Full text of the second candidate solution.

    Returns:
        Rendered prompt string for the merger model.

    """
    return CLEAN_MERGE_PROMPT_TEMPLATE.format(
        problem=problem,
        solution_1=solution_1,
        solution_2=solution_2,
    )
