import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset


@dataclass
class Sample:
    sid: str
    text: str          
    schema: str         
    reference: str    


# toy dataset as fallback
TOY_DATASET = [
    {
        "id": "person-1",
        "text": "John Smith is a 35-year-old software engineer from San Francisco. He has been working at TechCorp for 8 years and specializes in machine learning.",
        "schema": "name, age, occupation, city, company, years_experience, specialty",
        "reference": {
            "name": "John Smith",
            "age": 35,
            "occupation": "software engineer",
            "city": "San Francisco",
            "company": "TechCorp",
            "years_experience": 8,
            "specialty": "machine learning"
        }
    },
    {
        "id": "person-2",
        "text": "Dr. Maria Garcia, aged 42, is a cardiologist at Boston General Hospital. She graduated from Harvard Medical School and has published over 50 research papers.",
        "schema": "name, age, occupation, workplace, education, publications",
        "reference": {
            "name": "Maria Garcia",
            "age": 42,
            "occupation": "cardiologist",
            "workplace": "Boston General Hospital",
            "education": "Harvard Medical School",
            "publications": 50
        }
    },
    {
        "id": "place-1",
        "text": "The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 330 meters tall. It attracts approximately 7 million visitors annually.",
        "schema": "name, city, country, year_built, height_meters, annual_visitors",
        "reference": {
            "name": "Eiffel Tower",
            "city": "Paris",
            "country": "France",
            "year_built": 1889,
            "height_meters": 330,
            "annual_visitors": 7000000
        }
    },
    {
        "id": "place-2",
        "text": "Central Park spans 843 acres in Manhattan, New York City. It was designed by Frederick Law Olmsted and opened in 1858. The park features 21 playgrounds and 36 bridges.",
        "schema": "name, size_acres, location, designer, year_opened, playgrounds, bridges",
        "reference": {
            "name": "Central Park",
            "size_acres": 843,
            "location": "Manhattan, New York City",
            "designer": "Frederick Law Olmsted",
            "year_opened": 1858,
            "playgrounds": 21,
            "bridges": 36
        }
    },
    {
        "id": "product-1",
        "text": "The iPhone 15 Pro is manufactured by Apple and retails for $999. It features a 6.1-inch display, 256GB storage, and an A17 Pro chip. Available in titanium finish.",
        "schema": "name, manufacturer, price_usd, display_inches, storage_gb, processor, finish",
        "reference": {
            "name": "iPhone 15 Pro",
            "manufacturer": "Apple",
            "price_usd": 999,
            "display_inches": 6.1,
            "storage_gb": 256,
            "processor": "A17 Pro",
            "finish": "titanium"
        }
    },
    {
        "id": "product-2",
        "text": "Sony WH-1000XM5 wireless headphones cost $349 and offer 30 hours of battery life. They feature active noise cancellation and weigh only 250 grams.",
        "schema": "name, brand, price_usd, battery_hours, noise_cancellation, weight_grams",
        "reference": {
            "name": "WH-1000XM5",
            "brand": "Sony",
            "price_usd": 349,
            "battery_hours": 30,
            "noise_cancellation": True,
            "weight_grams": 250
        }
    },
    {
        "id": "person-3",
        "text": "Emily Chen, 28, works as a data analyst at DataFlow Inc in Seattle. She holds a Master's degree in Statistics and earns an annual salary of $95,000.",
        "schema": "name, age, occupation, company, city, degree, salary_usd",
        "reference": {
            "name": "Emily Chen",
            "age": 28,
            "occupation": "data analyst",
            "company": "DataFlow Inc",
            "city": "Seattle",
            "degree": "Master's in Statistics",
            "salary_usd": 95000
        }
    },
    {
        "id": "place-3",
        "text": "The Grand Canyon National Park in Arizona covers 1,217,262 acres. It was established in 1919 and receives about 6 million visitors per year. The canyon is up to 18 miles wide.",
        "schema": "name, state, size_acres, year_established, annual_visitors, max_width_miles",
        "reference": {
            "name": "Grand Canyon National Park",
            "state": "Arizona",
            "size_acres": 1217262,
            "year_established": 1919,
            "annual_visitors": 6000000,
            "max_width_miles": 18
        }
    },
    {
        "id": "product-3",
        "text": "The Tesla Model 3 is an electric vehicle with a range of 272 miles. It accelerates from 0-60 mph in 5.8 seconds and has a starting price of $38,990. Seats 5 passengers.",
        "schema": "name, type, range_miles, acceleration_0_60, price_usd, seating_capacity",
        "reference": {
            "name": "Tesla Model 3",
            "type": "electric vehicle",
            "range_miles": 272,
            "acceleration_0_60": 5.8,
            "price_usd": 38990,
            "seating_capacity": 5
        }
    },
    {
        "id": "person-4",
        "text": "Chef Antonio Rossi, 55, owns three Italian restaurants in Chicago. He trained in Rome for 10 years and has won 2 Michelin stars. His signature dish is handmade pasta.",
        "schema": "name, age, occupation, num_restaurants, city, training_location, training_years, michelin_stars, signature_dish",
        "reference": {
            "name": "Antonio Rossi",
            "age": 55,
            "occupation": "chef",
            "num_restaurants": 3,
            "city": "Chicago",
            "training_location": "Rome",
            "training_years": 10,
            "michelin_stars": 2,
            "signature_dish": "handmade pasta"
        }
    },
]


def load_samples(cfg: Dict[str, Any]) -> List[Sample]:
    """
    Load JSON extraction samples.
    
    Supports multiple sources:
    - "toy": Use built-in toy dataset (10 samples) - clean, reliable ground truth
    - "ner": Use CoNLL-2003 NER dataset from HuggingFace (entities extraction)
    - "json_struct": Use MasterControlAIML/JSON-Unstructured-Structured from HuggingFace
    """
    dataset_cfg = cfg.get("dataset", {})
    source = dataset_cfg.get("source", "toy")
    n = int(dataset_cfg.get("n_samples", 10))
    
    if source == "toy":
        return _load_toy_samples(n)
    elif source == "ner":
        return _load_ner_samples(n)
    elif source == "json_struct":
        return _load_json_struct_samples(n)
    else:
        raise ValueError(f"json_extraction supports source: toy, ner, json_struct. Got: {source}")


def _load_toy_samples(n: int) -> List[Sample]:
    """Load from built-in toy dataset."""
    samples: List[Sample] = []
    for i, item in enumerate(TOY_DATASET):
        if i >= n:
            break
        samples.append(Sample(
            sid=item["id"],
            text=item["text"],
            schema=item["schema"],
            reference=json.dumps(item["reference"], indent=2),
        ))
    return samples


def _load_json_struct_samples(n: int) -> List[Sample]:
    """
    Load from MasterControlAIML/JSON-Unstructured-Structured dataset.
    
    This dataset contains text with expected JSON structure output.
    Falls back to toy dataset if loading fails.
    """
    try:
        dataset = load_dataset(
            "MasterControlAIML/JSON-Unstructured-Structured", 
            split="train",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Warning: Could not load JSON-Unstructured-Structured dataset: {e}")
        print("Falling back to toy dataset...")
        return _load_toy_samples(n)
    
    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if len(samples) >= n:
            break
        
        try:
            # the dataset has 'unstructured_text' and 'structured_json' fields
            text = item.get("unstructured_text", item.get("text", ""))
            structured = item.get("structured_json", item.get("json", ""))
            
            if not text or not structured:
                continue
            
            # parse the structured JSON to extract schema
            if isinstance(structured, str):
                try:
                    parsed = json.loads(structured)
                except json.JSONDecodeError:
                    continue
            else:
                parsed = structured
            
            # extract schema from keys
            if isinstance(parsed, dict):
                schema = ", ".join(parsed.keys())
                reference = json.dumps(parsed, indent=2)
            else:
                continue
            
            # skip if text is too long (>500 chars) for reasonable inference
            if len(text) > 500:
                continue
            
            samples.append(Sample(
                sid=f"json-struct-{i}",
                text=text,
                schema=schema,
                reference=reference,
            ))
        except Exception:
            continue
    
    # if we didn't get enough samples, supplement with toy data
    if len(samples) < n:
        print(f"Only got {len(samples)} samples from HuggingFace, supplementing with toy data...")
        toy_samples = _load_toy_samples(n - len(samples))
        samples.extend(toy_samples)
    
    return samples


def _load_ner_samples(n: int) -> List[Sample]:
    """
    Load from CoNLL-2003 NER dataset.
    
    Task: Extract named entities (persons, organizations, locations) from text.
    Falls back to toy dataset if HuggingFace dataset fails.
    """
    # try to load CoNLL-2003 dataset
    try:
        dataset = load_dataset("conll2003", split="test")
    except Exception as e1:
        try:
            # try alternate source
            dataset = load_dataset("eriktks/conll2003", split="test")
        except Exception as e2:
            print(f"Warning: Could not load CoNLL-2003 dataset, falling back to toy data. Error: {e2}")
            return _load_toy_samples(n)
    
    # nER tag mapping for CoNLL-2003
    # tags: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
    tag_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    
    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if i >= n:
            break
        
        tokens = item["tokens"]
        ner_tags = item["ner_tags"]
        
        # reconstruct text
        text = " ".join(tokens)
        
        # extract entities
        entities = {"persons": [], "organizations": [], "locations": [], "misc": []}
        current_entity = []
        current_type = None
        
        for token, tag_id in zip(tokens, ner_tags):
            tag = tag_names[tag_id]
            
            if tag.startswith("B-"):
                # ave previous entity if exists
                if current_entity and current_type:
                    entity_text = " ".join(current_entity)
                    if current_type == "PER":
                        entities["persons"].append(entity_text)
                    elif current_type == "ORG":
                        entities["organizations"].append(entity_text)
                    elif current_type == "LOC":
                        entities["locations"].append(entity_text)
                    else:
                        entities["misc"].append(entity_text)
                
                # start new entity
                current_entity = [token]
                current_type = tag[2:]  # remove "B-" prefix
            elif tag.startswith("I-") and current_type == tag[2:]:
                # continue current entity
                current_entity.append(token)
            else:
                # end current entity
                if current_entity and current_type:
                    entity_text = " ".join(current_entity)
                    if current_type == "PER":
                        entities["persons"].append(entity_text)
                    elif current_type == "ORG":
                        entities["organizations"].append(entity_text)
                    elif current_type == "LOC":
                        entities["locations"].append(entity_text)
                    else:
                        entities["misc"].append(entity_text)
                current_entity = []
                current_type = None
        
        # don't forget last entity
        if current_entity and current_type:
            entity_text = " ".join(current_entity)
            if current_type == "PER":
                entities["persons"].append(entity_text)
            elif current_type == "ORG":
                entities["organizations"].append(entity_text)
            elif current_type == "LOC":
                entities["locations"].append(entity_text)
            else:
                entities["misc"].append(entity_text)
        
        # skip samples with no entities
        if not any(entities.values()):
            continue
        
        samples.append(Sample(
            sid=f"conll-{i}",
            text=text,
            schema="persons, organizations, locations, misc",
            reference=json.dumps(entities, indent=2),
        ))
        
        if len(samples) >= n:
            break
    
    return samples


def extract_json_from_prediction(prediction: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from model prediction.
    
    Tries multiple strategies:
    1. Parse the entire response as JSON
    2. Find JSON block in markdown code fence
    3. Find JSON object pattern { ... }
    """
    prediction = prediction.strip()
    
    # strategy 1: Try parsing the entire response
    try:
        return json.loads(prediction)
    except json.JSONDecodeError:
        pass
    
    # strategy 2: Look for JSON in markdown code block
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", prediction, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # strategy 3: Find JSON object pattern
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", prediction, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def _normalize_value(val) -> str:
    """Normalize a value for comparison (lowercase, strip whitespace)."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return str(val).lower()
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        return val.lower().strip()
    if isinstance(val, list):
        return str(sorted([_normalize_value(v) for v in val]))
    if isinstance(val, dict):
        return str({k: _normalize_value(v) for k, v in sorted(val.items())})
    return str(val).lower().strip()


def _values_match(pred_val, ref_val) -> bool:
    """Check if two values match (with some tolerance)."""
    # normalize both values
    pred_norm = _normalize_value(pred_val)
    ref_norm = _normalize_value(ref_val)
    
    # exact match after normalization
    if pred_norm == ref_norm:
        return True
    
    # for strings, check if one contains the other (handles "Dr. Maria Garcia" vs "Maria Garcia")
    if isinstance(ref_val, str) and isinstance(pred_val, str):
        ref_lower = ref_val.lower().strip()
        pred_lower = pred_val.lower().strip()
        if ref_lower in pred_lower or pred_lower in ref_lower:
            return True
    
    # for numbers, allow small tolerance
    if isinstance(ref_val, (int, float)) and isinstance(pred_val, (int, float)):
        if ref_val == 0:
            return pred_val == 0
        return abs(pred_val - ref_val) / abs(ref_val) < 0.01  # 1% tolerance
    
    return False


def accuracy_check(prediction: str, reference: str) -> bool:
    """
    Check if the prediction contains valid JSON with correct field values.
    
    Accuracy criteria (STRICT - to differentiate model quality):
    1. Must produce valid JSON
    2. Must have all required fields
    3. At least 90% of field values must match EXACTLY (stricter threshold)
    
    Note: The toy dataset is relatively easy (explicit facts in text).
    Use stricter matching to better differentiate model quality.
    For harder evaluation, use source: "ner" or "json_struct" in config.yaml.
    
    Args:
        prediction: The model's full response text
        reference: The expected JSON string
    
    Returns:
        True if valid JSON with >= 90% correct field values, False otherwise
    """
    # parse the reference to get expected fields
    try:
        ref_dict = json.loads(reference)
    except json.JSONDecodeError:
        return False
    
    # extract JSON from prediction
    pred_dict = extract_json_from_prediction(prediction)
    
    if pred_dict is None:
        return False
    
    # check if all required fields are present
    required_fields = set(ref_dict.keys())
    present_fields = set(pred_dict.keys())
    
    # all required fields must be present
    if not required_fields.issubset(present_fields):
        return False
    
    # count matching values - use STRICT matching
    matches = 0
    total = len(ref_dict)
    
    for field, ref_val in ref_dict.items():
        pred_val = pred_dict.get(field)
        if _values_match_strict(pred_val, ref_val):
            matches += 1
    
    # require at least 90% of values to match exactly
    return (matches / total) >= 0.90


def _values_match_strict(pred_val, ref_val) -> bool:
    """
    STRICT value matching - less forgiving than _values_match.
    
    This helps differentiate model quality on the toy dataset.
    """
    # normalize both values
    pred_norm = _normalize_value(pred_val)
    ref_norm = _normalize_value(ref_val)
    
    # exact match after normalization
    if pred_norm == ref_norm:
        return True
    
    # for strings, require exact match or exact substring (no partial)
    if isinstance(ref_val, str) and isinstance(pred_val, str):
        ref_lower = ref_val.lower().strip()
        pred_lower = pred_val.lower().strip()
        # only allow if prediction exactly equals reference (case-insensitive)
        # or if one is a title variant (Dr., Mr., etc.)
        if ref_lower == pred_lower:
            return True
        # allow "Dr. Maria Garcia" to match "Maria Garcia" but not vice versa
        if pred_lower.replace("dr. ", "").replace("mr. ", "").replace("ms. ", "") == ref_lower:
            return True
        if ref_lower.replace("dr. ", "").replace("mr. ", "").replace("ms. ", "") == pred_lower:
            return True
        return False
    
    # for numbers, require exact match (no tolerance)
    if isinstance(ref_val, (int, float)) and isinstance(pred_val, (int, float)):
        # allow int/float type differences (35 == 35.0)
        return float(pred_val) == float(ref_val)
    
    # for booleans
    if isinstance(ref_val, bool) and isinstance(pred_val, bool):
        return ref_val == pred_val
    
    return False
