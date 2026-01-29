import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset


@dataclass
class Sample:
    sid: str
    puzzle: str       # the logic puzzle/problem
    puzzle_type: str  # type of reasoning required
    reference: str    # the correct answer


# toy dataset as fallback
TOY_DATASET = [
    # sequence puzzles
    {
        "id": "seq-1",
        "type": "sequence",
        "puzzle": "What comes next in this sequence? 2, 6, 12, 20, 30, ?",
        "reference": "42",
        "explanation": "Pattern: differences are 4, 6, 8, 10, 12 (increasing by 2). Next: 30 + 12 = 42"
    },
    {
        "id": "seq-2", 
        "type": "sequence",
        "puzzle": "What is the next number in this sequence? 1, 1, 2, 3, 5, 8, 13, ?",
        "reference": "21",
        "explanation": "Fibonacci sequence: each number is the sum of the two preceding ones. 8 + 13 = 21"
    },
    {
        "id": "seq-3",
        "type": "sequence",
        "puzzle": "Complete the pattern: 3, 9, 27, 81, ?",
        "reference": "243",
        "explanation": "Each number is multiplied by 3. 81 × 3 = 243"
    },
    
    # pattern recognition
    {
        "id": "pat-1",
        "type": "pattern",
        "puzzle": "If A=1, B=2, C=3, and so on, what is the sum of the letters in the word 'CAT'?",
        "reference": "24",
        "explanation": "C=3, A=1, T=20. Sum = 3 + 1 + 20 = 24"
    },
    {
        "id": "pat-2",
        "type": "pattern",
        "puzzle": "In a code, APPLE is written as ELPPA. How would ORANGE be written in the same code?",
        "reference": "EGNARO",
        "explanation": "The code reverses the letters. ORANGE reversed is EGNARO"
    },
    
    # deductive reasoning
    {
        "id": "ded-1",
        "type": "deductive",
        "puzzle": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
        "reference": "No",
        "explanation": "This is a logical fallacy. Just because some flowers fade quickly doesn't mean any roses do."
    },
    {
        "id": "ded-2",
        "type": "deductive",
        "puzzle": "If all doctors are professionals, and all professionals have degrees, what can we conclude about doctors?",
        "reference": "All doctors have degrees",
        "explanation": "Transitive logic: doctors → professionals → degrees, so doctors → degrees"
    },
    {
        "id": "ded-3",
        "type": "deductive",
        "puzzle": "Tom is taller than Jerry. Jerry is taller than Spike. Who is the shortest?",
        "reference": "Spike",
        "explanation": "Tom > Jerry > Spike, so Spike is shortest"
    },
    
    # mathematical reasoning
    {
        "id": "math-1",
        "type": "mathematical",
        "puzzle": "A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost in cents?",
        "reference": "5",
        "explanation": "Let ball = x. Bat = x + 100. Total: x + (x + 100) = 110. 2x = 10, x = 5 cents"
    },
    {
        "id": "math-2",
        "type": "mathematical",
        "puzzle": "If 5 machines take 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?",
        "reference": "5",
        "explanation": "Each machine makes 1 widget in 5 minutes. With 100 machines making 100 widgets (1 each), it still takes 5 minutes."
    },
]


def load_samples(cfg: Dict[str, Any]) -> List[Sample]:
    """
    Load logical reasoning samples.
    
    Supports multiple sources:
    - "toy": Use built-in toy dataset (10 puzzles)
    - "logiqa": Use LogiQA dataset (logical reasoning multiple choice)
    - "boolq": Use BoolQ dataset (yes/no reasoning questions)
    """
    dataset_cfg = cfg.get("dataset", {})
    source = dataset_cfg.get("source", "toy")
    n = int(dataset_cfg.get("n_samples", 10))
    
    if source == "toy":
        return _load_toy_samples(n)
    elif source == "logiqa":
        return _load_logiqa_samples(n)
    elif source == "boolq":
        return _load_boolq_samples(n)
    else:
        raise ValueError(f"reasoning supports source: toy, logiqa, boolq. Got: {source}")


def _load_toy_samples(n: int) -> List[Sample]:
    """Load from built-in toy dataset."""
    samples: List[Sample] = []
    for i, item in enumerate(TOY_DATASET):
        if i >= n:
            break
        samples.append(Sample(
            sid=item["id"],
            puzzle=item["puzzle"],
            puzzle_type=item["type"],
            reference=item["reference"],
        ))
    return samples


def _load_logiqa_samples(n: int) -> List[Sample]:
    """
    Load from LogiQA dataset.
    
    LogiQA is a logical reasoning dataset with multiple choice questions
    derived from the Chinese Civil Service Examination.
    """
    dataset = load_dataset("lucasmccabe/logiqa", split="test", trust_remote_code=True)
    
    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if len(samples) >= n:
            break
        
        context = item["context"]
        question = item["query"]
        options = item["options"]
        label = item["correct_option"]  # 0-3 index
        
        # format as multiple choice
        options_text = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(options)])
        puzzle = f"{context}\n\nQuestion: {question}\n\nOptions:\n{options_text}\n\nAnswer with just the letter (A, B, C, or D)."
        
        # reference is the correct letter
        reference = chr(65 + label)
        
        samples.append(Sample(
            sid=f"logiqa-{i}",
            puzzle=puzzle,
            puzzle_type="logical_reasoning",
            reference=reference,
        ))
    
    return samples


def _load_boolq_samples(n: int) -> List[Sample]:
    """
    Load from BoolQ dataset.
    
    BoolQ is a yes/no question answering dataset from Google.
    Questions require reading comprehension and reasoning.
    """
    dataset = load_dataset("google/boolq", split="validation", trust_remote_code=True)
    
    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if len(samples) >= n:
            break
        
        passage = item["passage"]
        question = item["question"]
        answer = item["answer"]  # true/False
        
        puzzle = f"Passage: {passage}\n\nQuestion: {question}\n\nAnswer with just 'Yes' or 'No'."
        reference = "Yes" if answer else "No"
        
        samples.append(Sample(
            sid=f"boolq-{i}",
            puzzle=puzzle,
            puzzle_type="boolean_reasoning",
            reference=reference,
        ))
    
    return samples


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.
    - Lowercase
    - Strip whitespace
    - Remove common prefixes like "the answer is"
    """
    answer = answer.lower().strip()
    
    # remove common answer prefixes
    prefixes = [
        "the answer is",
        "answer:",
        "answer is",
        "the final answer is",
        "final answer:",
        "therefore,",
        "so,",
        "thus,",
    ]
    
    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    # remove trailing punctuation
    answer = answer.rstrip(".,!?")
    
    return answer


def extract_answer_from_prediction(prediction: str) -> Optional[str]:
    """
    Extract the final answer from a model's prediction.
    
    Tries multiple strategies:
    1. Look for "#### answer" format
    2. Look for "answer is X" or "answer: X" patterns
    3. Look for boxed answers
    4. Look for the last line/sentence
    """
    prediction = prediction.strip()
    
    # strategy 1: GSM8K-style "#### answer" format
    match = re.search(r"####\s*(.+?)$", prediction, re.MULTILINE)
    if match:
        return match.group(1).strip()
    
    # strategy 2: "the answer is X" or "answer: X" patterns
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+([^\n.]+)",
        r"(?:the\s+)?(?:final\s+)?answer[:\s]+([^\n.]+)",
        r"therefore[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\n.]+)",
        r"thus[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\n.]+)",
        r"so[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\n.]+)",
        r"conclusion[:\s]+([^\n.]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, prediction, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # strategy 3: LaTeX boxed format
    match = re.search(r"\\boxed\{([^}]+)\}", prediction)
    if match:
        return match.group(1).strip()
    
    # strategy 4: Bold markdown answer
    match = re.search(r"\*\*([^*]+)\*\*\s*$", prediction, re.MULTILINE)
    if match:
        return match.group(1).strip()
    
    # strategy 5: Last line that looks like an answer
    lines = prediction.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and len(line) < 100 and not line.startswith('#'):
            # check if it's a standalone answer-like line
            if re.match(r"^[\w\s\-\',]+$", line) or re.match(r"^\d+$", line):
                return line
    
    return None


def accuracy_check(prediction: str, reference: str) -> bool:
    """
    Check if the prediction's final answer matches the reference.
    
    Args:
        prediction: The model's full response text
        reference: The correct answer
    
    Returns:
        True if the answer matches, False otherwise
    """
    # extract the answer from the prediction
    pred_answer = extract_answer_from_prediction(prediction)
    
    if pred_answer is None:
        # fallback: check if reference appears in prediction
        return normalize_answer(reference) in normalize_answer(prediction)
    
    # normalize both for comparison
    pred_normalized = normalize_answer(pred_answer)
    ref_normalized = normalize_answer(reference)
    
    # exact match
    if pred_normalized == ref_normalized:
        return True
    
    # check if one contains the other (for answers like "5 cents" vs "5")
    if ref_normalized in pred_normalized or pred_normalized in ref_normalized:
        return True
    
    # try numeric comparison for number answers
    try:
        # extract numbers from both
        pred_nums = re.findall(r'-?\d+(?:\.\d+)?', pred_normalized)
        ref_nums = re.findall(r'-?\d+(?:\.\d+)?', ref_normalized)
        
        if pred_nums and ref_nums:
            if float(pred_nums[-1]) == float(ref_nums[-1]):
                return True
    except (ValueError, IndexError):
        pass
    
    return False
