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

import os
import pickle
import time
from typing import Any, Callable, Dict, Optional, Tuple


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        prefix: str,
        checkpoint_every: Optional[int] = None,
        resume: bool = True,
    ):
        self.checkpoint_dir = checkpoint_dir or "."
        self.prefix = prefix
        self.resume_enabled = resume
        self.checkpoint_every = checkpoint_every
        self.eval_count = 0
        self.counts_by_key: Dict[Any, int] = {}
        self.skip_remaining_by_key: Dict[Any, int] = {}
        self._last_checkpoint_eval_count = 0
        self._resumed = False

    def _checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.prefix}latest.pkl")

    def _meta_path(self, checkpoint_path: str) -> str:
        return f"{checkpoint_path}.meta.pkl"

    def find_latest_checkpoint(self) -> Optional[str]:
        if not os.path.isdir(self.checkpoint_dir):
            return None
        checkpoint_path = self._checkpoint_path()
        if os.path.isfile(checkpoint_path):
            return checkpoint_path
        return None

    def load_latest(self) -> Optional[Tuple[Any, Dict[str, Any], str]]:
        if not self.resume_enabled or self._resumed:
            return None
        checkpoint_path = self.find_latest_checkpoint()
        if not checkpoint_path:
            self._resumed = True
            return None

        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)

        meta_path = self._meta_path(checkpoint_path)
        meta: Dict[str, Any] = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

        self._resumed = True
        return results, meta, checkpoint_path

    def resume_from_checkpoint(
        self,
        count_key: str,
        count_fallback_fn: Callable[[Any], Dict[Any, int]],
    ) -> Optional[Tuple[Any, Dict[str, Any], str]]:
        loaded = self.load_latest()
        if not loaded:
            self._resumed = True
            return None

        results, meta, checkpoint_path = loaded
        counts_by_key = meta.get(count_key, {})
        eval_count = meta.get("eval_count", 0)

        counts_by_key = count_fallback_fn(results)
        eval_count = sum(counts_by_key.values())
        if not eval_count:
            eval_count = sum(counts_by_key.values())

        self.eval_count = eval_count
        self.counts_by_key = counts_by_key
        self.skip_remaining_by_key = dict(counts_by_key)
        self._resumed = True
        return results, meta, checkpoint_path

    def increment(self, key: Any, count: int = 1, **kwargs):
        if count <= 0:
            return
        self.eval_count += count
        self.counts_by_key[key] = self.counts_by_key.get(key, 0) + count

    def save(self, results: Any, meta: Dict[str, Any]) -> str:
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            checkpoint_path = self._checkpoint_path()
            with open(checkpoint_path, "wb") as f:
                pickle.dump(results, f)
            with open(self._meta_path(checkpoint_path), "wb") as f:
                pickle.dump(meta, f)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return None
        return checkpoint_path

    def save_checkpoint(self, results: Any, extra_meta: Dict[str, Any] = {}):
        meta = {"eval_count": self.eval_count}
        # meta.update(extra_meta or {})
        self.save(results, meta)

    def checkpoint_if_due(self, results: Any, extra_meta: Dict[str, Any] = None):
        if not self.checkpoint_every:
            return
        if self.eval_count <= 0:
            return
        if self.eval_count == self._last_checkpoint_eval_count:
            return
        if self.eval_count % self.checkpoint_every == 0:
            self._last_checkpoint_eval_count = self.eval_count
            self.save_checkpoint(results, extra_meta or {})

    def cleanup(self):
        """Remove the checkpoint and meta files managed by this instance."""
        checkpoint_path = self._checkpoint_path()
        meta_path = self._meta_path(checkpoint_path)
        for path in [checkpoint_path, meta_path]:
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"Warning: Could not remove checkpoint file {path}: {e}")

    @staticmethod
    def cleanup_by_prefix(checkpoint_dir, prefix):
        """Remove all checkpoint files whose name starts with the given prefix."""
        checkpoint_dir = checkpoint_dir or "."
        if not os.path.isdir(checkpoint_dir):
            return
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith(prefix):
                filepath = os.path.join(checkpoint_dir, filename)
                if os.path.isfile(filepath):
                    try:
                        os.remove(filepath)
                    except OSError as e:
                        print(f"Warning: Could not remove {filepath}: {e}")
