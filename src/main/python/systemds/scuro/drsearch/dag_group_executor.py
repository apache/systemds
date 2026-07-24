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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import pickle

import time
from typing import Any, Dict, List, Optional
from systemds.scuro import Modality
from systemds.scuro.drsearch.representation_dag import (
    LRUCache,
    RepresentationDag,
    group_dags_by_dependencies,
)
from systemds.scuro.utils.checkpointing import CheckpointManager
from systemds.scuro.drsearch.dag_group_scheduler import DAGGroupScheduler


def _process_dag_group(
    dag_group_pickle: bytes,
    modality_pickle: bytes,
    tasks_pickle: bytes,
    modality_id: int,
    dag_group_idx: int,
) -> Dict[str, Any]:
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=os.getcwd(),
        prefix=f"unimodal_checkpoint_group_{modality_id}_{dag_group_idx}_",
        checkpoint_every=1,
        resume=False,
    )
    results = []

    dag_group = pickle.loads(dag_group_pickle)
    modality = pickle.loads(modality_pickle)
    tasks = pickle.loads(tasks_pickle)

    group_cache = LRUCache(max_size=6)

    for i, dag in enumerate(dag_group):
        representation = dag.execute([modality], external_cache=group_cache)

        for task in tasks:
            start = time.perf_counter()
            scores = task.run(representation.data)
            end = time.perf_counter()

            results.append(
                {
                    "scores": scores,
                    "transform_time": representation.transform_time,
                    "task_name": task.model.name,
                    "task_time": end - start,
                    "dag": dag,
                    "modality_id": modality_id,
                }
            )

            checkpoint_manager.increment(modality_id, 1, dag_group_idx=dag_group_idx)
            checkpoint_manager.checkpoint_if_due(results)

    return {"results": results}


class DAGGroupExecutor:
    def __init__(
        self,
        dags: List[RepresentationDag],
        modalities: List[Modality],
        tasks: List[Any],
        checkpoint_manager: Optional[CheckpointManager] = None,
        max_workers: Optional[int] = None,
    ):
        self.dags = dags
        self.dag_groups = group_dags_by_dependencies(dags)
        self.modalities = modalities
        self.tasks = tasks
        self.max_workers = max_workers or mp.cpu_count()
        self.checkpoint_manager = checkpoint_manager
        self.scheduler = DAGGroupScheduler(
            dag_groups=self.dag_groups, modality=modalities[0]
        )

    def run(self):
        results = []
        ctx = mp.get_context("spawn")
        max_workers = min(len(self.dag_groups), self.max_workers)

        modality_pickle = pickle.dumps(
            self.modalities[0]
        )  # TODO: handle multiple modalities
        tasks_pickle = pickle.dumps(self.tasks)

        pending_dag_groups = set(range(len(self.dag_groups)))
        running_dag_groups = {}
        all_groups_succeeded = True
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            while pending_dag_groups or running_dag_groups:
                pending_resources = [
                    (
                        i,
                        self.scheduler.group_resources[i][0],
                        self.scheduler.group_resources[i][1],
                    )
                    for i in pending_dag_groups
                ]
                ready_to_execute = self.scheduler.get_runnable(
                    pending_resources, max_concurrent=max_workers
                )
                for group_id, gpu_id in ready_to_execute:
                    pending_dag_groups.remove(group_id)
                    dag_group = self.dag_groups[group_id]
                    cpu_mem, gpu_mem = self.scheduler.group_resources[group_id]

                    future = executor.submit(
                        _process_dag_group,
                        pickle.dumps(dag_group),
                        modality_pickle,
                        tasks_pickle,
                        self.modalities[0].modality_id,
                        group_id,
                    )
                    running_dag_groups[future] = (group_id, cpu_mem, gpu_mem, gpu_id)
                if not running_dag_groups:
                    break
                done = next(as_completed(running_dag_groups), None)
                if done is None:
                    break
                group_id, cpu_mem, gpu_mem, gpu_id = running_dag_groups.pop(done)
                self.scheduler.release(cpu_mem, gpu_mem, gpu_id)

            try:
                result_dict = future.result()

                for result_entry in result_dict["results"]:
                    results.append(
                        {
                            "scores": result_entry["scores"],
                            "transform_time": result_entry["transform_time"],
                            "task_name": result_entry["task_name"],
                            "task_time": result_entry["task_time"],
                            "dag": result_entry["dag"],
                            "modality_id": self.modalities[0].modality_id,
                        }
                    )
            except Exception as e:
                all_groups_succeeded = False
                print(
                    f"Error processing DAG group {group_id} for modality {self.modalities[0].modality_id}: {e}"
                )
        return results, all_groups_succeeded
