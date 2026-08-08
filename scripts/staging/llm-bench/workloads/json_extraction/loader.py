#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    sid: str
    text: str
    schema: str
    reference: str


def load_samples(cfg: Dict[str, Any]) -> List[Sample]:
    dataset_cfg = cfg.get("dataset", {})
    source = dataset_cfg.get("source", "ner")
    n = int(dataset_cfg.get("n_samples", 10))

    if source == "ner":
        samples = _load_ner_samples(n)
    else:
        raise ValueError(f"json_extraction supports source: ner. Got: {source}")

    if len(samples) < n:
        logger.warning("Requested %d samples but only %d available (source=%s)", n, len(samples), source)
    return samples


def _load_ner_samples(n: int) -> List[Sample]:
    # try to load CoNLL-2003 dataset
    try:
        dataset = load_dataset("conll2003", split="test")
    except Exception as e1:
        try:
            # try alternate source
            dataset = load_dataset("eriktks/conll2003", split="test")
        except Exception as e2:
            raise RuntimeError(
                f"Could not load CoNLL-2003 NER dataset from HuggingFace. "
                f"Primary error: {e1}  |  Alternate error: {e2}  |  "
                f"Check your internet connection or install the dataset manually."
            ) from e2

    # CoNLL-2003 BIO tags
    tag_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if len(samples) >= n:
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
                # save previous entity if exists
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
    prediction = prediction.strip()

    # try parsing whole response
    try:
        return json.loads(prediction)
    except json.JSONDecodeError:
        pass

    # look inside ```json ... ``` blocks
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", prediction, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # find { ... } pattern
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", prediction, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _normalize_value(val) -> str:
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


def _compute_entity_metrics(pred_dict: Dict, ref_dict: Dict) -> Dict[str, float]:
    """Entity-level P/R/F1 across all list-valued fields."""
    total_correct = 0
    total_pred = 0
    total_ref = 0

    for field, ref_val in ref_dict.items():
        if not isinstance(ref_val, list):
            continue
        pred_val = pred_dict.get(field, [])
        if not isinstance(pred_val, list):
            pred_val = []

        ref_set = {_normalize_value(v) for v in ref_val}
        pred_set = {_normalize_value(v) for v in pred_val}

        total_correct += len(ref_set & pred_set)
        total_pred += len(pred_set)
        total_ref += len(ref_set)

    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_ref if total_ref > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "entities_correct": total_correct,
        "entities_predicted": total_pred,
        "entities_reference": total_ref,
    }


def accuracy_check(prediction: str, reference: str) -> bool:
    """NER: entity F1 >= 0.5 across all entity categories."""
    accuracy_check.last_entity_metrics = None

    try:
        ref_dict = json.loads(reference)
    except json.JSONDecodeError:
        return False

    pred_dict = extract_json_from_prediction(prediction)
    if pred_dict is None or not isinstance(pred_dict, dict):
        return False

    entity_metrics = _compute_entity_metrics(pred_dict, ref_dict)
    accuracy_check.last_entity_metrics = entity_metrics
    return entity_metrics["entity_f1"] >= 0.5
