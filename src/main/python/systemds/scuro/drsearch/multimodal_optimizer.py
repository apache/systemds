from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)

from systemds.scuro.representations.aggregate import Aggregation

from systemds.scuro.drsearch.operator_registry import Registry

from systemds.scuro.utils.schema_helpers import get_shape
import dataclasses


class MultimodalOptimizer:
    def __init__(self, modalities, unimodal_optimizer, tasks, k=2, debug=True):
        self.k_best_modalities = None
        self.modalities = modalities
        self.unimodal_optimizer = unimodal_optimizer
        self.tasks = tasks
        self.k = k
        self.extract_k_best_modalities_per_task()
        self.operator_registry = Registry()
        self.optimization_results = MultimodalResults(
            modalities, tasks, debug, self.k_best_modalities
        )
        self.cache = {}

    def optimize(self):
        for task in self.tasks:
            self.optimize_intramodal_representations(task)
            self.optimize_intermodal_representations(task)

    def optimize_intramodal_representations(self, task):
        for modality in self.modalities:
            representations = self.k_best_modalities[task.model.name][
                modality.modality_id
            ]
            applied_representations = self.extract_representations(
                representations, modality
            )

            for i in range(1, len(applied_representations)):
                for fusion_method in self.operator_registry.get_fusion_operators():
                    if fusion_method().needs_alignment and not applied_representations[
                        i - 1
                    ].is_aligned(applied_representations[i]):
                        continue
                    combined = applied_representations[i - 1].combine(
                        applied_representations[i], fusion_method()
                    )
                    self.evaluate(
                        task,
                        combined,
                        [i - 1, i],
                        fusion_method,
                        [
                            applied_representations[i - 1].modality_id,
                            applied_representations[i].modality_id,
                        ],
                    )
                    if not fusion_method().commutative:
                        combined_comm = applied_representations[i].combine(
                            applied_representations[i - 1], fusion_method()
                        )
                        self.evaluate(
                            task,
                            combined_comm,
                            [i, i - 1],
                            fusion_method,
                            [
                                applied_representations[i - 1].modality_id,
                                applied_representations[i].modality_id,
                            ],
                        )

    def optimize_intermodal_representations(self, task):
        pass

    def extract_representations(self, representations, modality):
        applied_representations = []
        for i in range(0, len(representations)):
            applied_representation = modality
            for j, rep in enumerate(representations[i].representations):
                representation, is_context = (
                    self.operator_registry.get_representation_by_name(
                        rep, modality.modality_type
                    )
                )
                if representation is None:
                    if rep == AggregatedRepresentation.__name__:
                        representation = AggregatedRepresentation(Aggregation())
                else:
                    representation = representation()
                representation.set_parameters(representations[i].params[j])
                if is_context:
                    applied_representation = applied_representation.context(
                        representation
                    )
                else:
                    applied_representation = (
                        applied_representation.apply_representation(representation)
                    )
            applied_representations.append(applied_representation)
        return applied_representations

    def evaluate(self, task, modality, representations, fusion, modality_ids):
        if task.expected_dim == 1 and get_shape(modality.metadata) > 1:
            for aggregation in Aggregation().get_aggregation_functions():
                # padding should not be necessary here
                agg_operator = AggregatedRepresentation(Aggregation(aggregation, False))
                agg_modality = agg_operator.transform(modality)

                scores = task.run(agg_modality.data)
                reps = representations.copy()
                reps.append(agg_operator)

                self.optimization_results.add_result(
                    scores, reps, [fusion], modality_ids, task.model.name
                )
        else:
            scores = task.run(modality.data)
            self.optimization_results.add_result(
                scores, representations, [fusion], modality_ids, task.model.name
            )

    def add_to_cache(self, result_idx, combined_modality):
        self.cache[result_idx] = combined_modality

    def extract_k_best_modalities_per_task(self):
        self.k_best_modalities = {}
        for task in self.tasks:
            self.k_best_modalities[task.model.name] = {}
            for modality in self.modalities:
                self.k_best_modalities[task.model.name][modality.modality_id] = (
                    self.unimodal_optimizer.get_k_best_results(modality, self.k, task)
                )


class MultimodalResults:
    def __init__(self, modalities, tasks, debug, k_best_modalities):
        self.modality_ids = [modality.modality_id for modality in modalities]
        self.task_names = [task.model.name for task in tasks]
        self.results = {}
        self.debug = debug
        self.k_best_modalities = k_best_modalities

        for task in tasks:
            self.results[task.model.name] = {}

    def add_result(
        self, scores, best_representation_idx, fusion_methods, modality_ids, task_name
    ):

        entry = MultimodalResultEntry(
            representations=best_representation_idx,
            train_score=scores[0],
            val_score=scores[1],
            fusion_methods=[fusion_method.__name__ for fusion_method in fusion_methods],
            modality_ids=modality_ids,
            task=task_name,
        )

        modality_id_strings = "_".join(list(map(str, modality_ids)))
        if not modality_id_strings in self.results[task_name]:
            self.results[task_name][modality_id_strings] = []

        self.results[task_name][modality_id_strings].append(entry)

    def print_results(self):
        for task_name in self.task_names:
            for modality in self.results[task_name].keys():
                for entry in self.results[task_name][modality]:
                    reps = []
                    for i, mod_idx in enumerate(entry.modality_ids):
                        reps.append(
                            self.k_best_modalities[task_name][mod_idx][
                                entry.representations[i]
                            ]
                        )

                    print(
                        f"{modality}_{task_name}: "
                        f"Validation score: {entry.val_score} - Training score: {entry.train_score}"
                    )
                    for i, rep in enumerate(reps):
                        print(
                            f"    Representation: {entry.modality_ids[i]} - {rep.representations}"
                        )
                        if i < len(reps) - 1:
                            print(f"    Fusion: {entry.fusion_methods[i]} ")

    def store_results(self, file_name=None):
        for task_name in self.task_names:
            for modality in self.results[task_name].keys():
                for entry in self.results[task_name][modality]:
                    reps = []
                    for i, mod_idx in enumerate(entry.modality_ids):
                        reps.append(
                            self.k_best_modalities[task_name][mod_idx][
                                entry.representations[i]
                            ]
                        )
                    entry.representations = reps

        import pickle

        if file_name is None:
            import time

            timestr = time.strftime("%Y%m%d-%H%M%S")
            file_name = "multimodal_optimizer" + timestr + ".pkl"

        with open(file_name, "wb") as f:
            pickle.dump(self.results, f)


@dataclasses.dataclass
class MultimodalResultEntry:
    val_score: float
    modality_ids: list
    representations: list
    fusion_methods: list
    train_score: float
    task: str
