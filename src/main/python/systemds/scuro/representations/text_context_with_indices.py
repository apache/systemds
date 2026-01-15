# -------------------------------------------------------------
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
# -------------------------------------------------------------
import re
from typing import List, Any

from systemds.scuro.drsearch.operator_registry import register_context_operator
from systemds.scuro.representations.context import Context
from systemds.scuro.modality.type import ModalityType

# TODO: Use this to get indices for text chunks based on different splitting strategies
# To use this approach a differnt extration of text chunks is needed in either the TextModality or the Representations


def _split_into_words(text: str) -> List[str]:
    """Split text into words, preserving whitespace structure."""
    if not text or not isinstance(text, str):
        return []
    return text.split()


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex.
    Handles common sentence endings: . ! ?
    """
    if not text or not isinstance(text, str):
        return []

    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])(?=\s*$)"
    sentences = re.split(sentence_pattern, text.strip())

    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text]

    return sentences


def _count_words(text: str) -> int:
    """
    Count the number of words in a text string.
    """
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


def _extract_text(instance: Any) -> str:
    if isinstance(instance, str):
        text = instance
    else:
        text = str(instance)

    if not text or not text.strip():
        return ""
    return text


# @register_context_operator(ModalityType.TEXT)
class WordCountSplitIndices(Context):
    """
    Splits text after a fixed number of words.

    Parameters:
        max_words (int): Maximum number of words per chunk (default: 55)
        overlap (int): Number of overlapping words between chunks (default: 0)
    """

    def __init__(self, max_words=55, overlap=0):
        parameters = {
            "max_words": [40, 50, 55, 60, 70, 250, 300, 350, 400, 450],
            "overlap": [0, 10, 20, 30],
        }
        super().__init__("WordCountSplit", parameters)
        self.max_words = int(max_words)
        self.overlap = max(0, int(overlap))

    def execute(self, modality):
        """
        Split each text instance into chunks of max_words words.

        Returns:
            List of tuples, where each tuple contains the start and end index of text chunks
        """
        chunked_data = []

        for instance in modality.data:
            text = _extract_text(instance)

            if not text:
                chunked_data.append((0, 0))
                continue

            words = _split_into_words(text)

            if len(words) <= self.max_words:
                chunked_data.append([(0, len(text))])
                continue

            chunks = []
            stride = self.max_words - self.overlap

            start = 0
            for i in range(0, len(words), stride):
                chunk_words = words[i : i + self.max_words]
                chunk_text = " ".join(chunk_words)
                chunks.append((start, start + len(chunk_text)))
                start += len(chunk_text) + 1

                if i + self.max_words >= len(words):
                    break

            chunked_data.append(chunks)

        return chunked_data


# @register_context_operator(ModalityType.TEXT)
class SentenceBoundarySplitIndices(Context):
    """
    Splits text at sentence boundaries while respecting maximum word count.

    Parameters:
        max_words (int): Maximum number of words per chunk (default: 55)
        min_words (int): Minimum number of words per chunk before splitting (default: 10)
    """

    def __init__(self, max_words=55, min_words=10, overlap=0.1):
        parameters = {
            "max_words": [40, 50, 55, 60, 70, 250, 300, 350, 400, 450],
            "min_words": [10, 20, 30],
        }
        super().__init__("SentenceBoundarySplit", parameters)
        self.max_words = int(max_words)
        self.min_words = max(1, int(min_words))
        self.overlap = overlap
        self.stride = max(1, int(max_words * (1 - overlap)))

    def execute(self, modality):
        """
        Split each text instance at sentence boundaries, respecting max_words.

        Returns:
            List of lists, where each inner list contains text chunks (strings)
        """
        chunked_data = []

        for instance in modality.data:
            text = _extract_text(instance)
            if not text:
                chunked_data.append((0, 0))
                continue

            sentences = _split_into_sentences(text)

            if not sentences:
                chunked_data.append((0, len(text)))
                continue

            chunks = []
            current_chunk = None
            current_word_count = 0
            start = 0
            for sentence in sentences:
                sentence_word_count = _count_words(sentence)

                if sentence_word_count > self.max_words:
                    if current_chunk and current_word_count >= self.min_words:
                        chunks.append(current_chunk)
                        current_chunk = []
                        current_word_count = 0

                    words = _split_into_words(sentence)
                    for i in range(0, len(words), self.max_words):
                        chunk_words = words[i : i + self.max_words]
                        current_chunk = (
                            (start, start + len(" ".join(chunk_words)))
                            if not current_chunk
                            else (current_chunk[0], start + len(" ".join(chunk_words)))
                        )
                        start += len(" ".join(chunk_words)) + 1

                elif current_word_count + sentence_word_count > self.max_words:
                    if current_chunk and current_word_count >= self.min_words:
                        chunks.append(current_chunk)
                        current_chunk = (start, start + len(sentence))
                        start += len(sentence) + 1
                        current_word_count = sentence_word_count
                    else:
                        current_chunk = (current_chunk[0], start + len(sentence))
                        start += len(sentence) + 1
                        current_word_count += sentence_word_count
                else:
                    current_chunk = (
                        (start, start + len(sentence))
                        if not current_chunk
                        else (current_chunk[0], start + len(sentence))
                    )
                    start += len(sentence) + 1
                    current_word_count += sentence_word_count

            # Add remaining chunk
            if current_chunk:
                chunks.append(current_chunk)

            if not chunks:
                chunks = [(0, len(text))]

            chunked_data.append(chunks)

        return chunked_data


# @register_context_operator(ModalityType.TEXT)
class OverlappingSplitIndices(Context):
    """
    Splits text with overlapping chunks using a sliding window approach.

    Parameters:
        max_words (int): Maximum number of words per chunk (default: 55)
        overlap (int): percentage of overlapping words between chunks (default: 50%)
        stride (int, optional): Step size in words. If None, stride = max_words - overlap_words
    """

    def __init__(self, max_words=55, overlap=0.5, stride=None):
        overlap_words = int(max_words * overlap)
        if stride is None:
            stride = max_words - overlap_words

        parameters = {
            "max_words": [40, 55, 70, 250, 300, 350, 400, 450],
            "overlap": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "stride": [10, 15, 20, 30],
        }
        super().__init__("OverlappingSplit", parameters)
        self.max_words = max_words
        self.overlap = overlap
        self.stride = stride

    def execute(self, modality):
        """
        Split each text instance with overlapping chunks.

        Returns:
            List of tuples, where each tuple contains start and end index to the text chunks
        """
        chunked_data = []

        for instance in modality.data:
            text = _extract_text(instance)
            if not text:
                chunked_data.append((0, 0))
                continue

            words = _split_into_words(text)

            if len(words) <= self.max_words:
                chunked_data.append((0, len(text)))
                continue

            chunks = []

            # Create overlapping chunks with specified stride
            start = 0
            for i in range(0, len(words), self.stride):
                chunk_words = words[i : i + self.max_words]
                if chunk_words:
                    chunk_text = " ".join(chunk_words)
                    chunks.append((start, start + len(chunk_text)))
                    start += len(chunk_text) - len(
                        " ".join(chunk_words[self.stride - len(chunk_words) :])
                    )
                if i + self.max_words >= len(words):
                    break

            if not chunks:
                chunks = [(0, len(text))]

            chunked_data.append(chunks)

        return chunked_data
