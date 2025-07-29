from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)

from systemds.scuro.representations.aggregate import Aggregation

from systemds.scuro.drsearch.operator_registry import Registry

from systemds.scuro.utils.schema_helpers import get_shape
import dataclasses


class MultimodalOptimizer:
    def __init__(self, modalities, unimodal_optimizer, tasks, k=2):
        self.k_best_modalities = None
        self.modalities = modalities
        self.unimodal_optimizer = unimodal_optimizer
        self.tasks = tasks
        self.k = k
        self.extract_k_best_modalities_per_task()
        self.operator_registry = Registry()
        self.optimization_results = {}

    def optimize(self):
        for task in self.tasks:
            for modality in self.modalities:
                representations = self.k_best_modalities[task][modality.modality_id]
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
                                applied_representation.apply_representation(
                                    representation
                                )
                            )
                    applied_representations.append(applied_representation)

    def evaluate(self, task, modality, representation_names, params):
        if task.expected_dim == 1 and get_shape(modality.metadata) > 1:
            for aggregation in Aggregation().get_aggregation_functions():
                # padding should not be necessary here
                agg_operator = AggregatedRepresentation(Aggregation(aggregation, False))
                agg_modality = agg_operator.transform(modality)

                scores = task.run(agg_modality.data)
                rep_names = representation_names.copy()
                rep_names.append(agg_operator.name)

                rep_params = params.copy()
                rep_params.append(agg_operator.parameters)
                self.optimization_results[modality.modality_id][
                    task.model.name
                ].add_result(scores, rep_names, rep_params)
        else:
            scores = task.run(modality.data)
            self.optimization_results[modality.modality_id][task.model.name].add_result(
                scores, representation_names, params
            )

    def extract_k_best_modalities_per_task(self):
        self.k_best_modalities = {}
        for task in self.tasks:
            self.k_best_modalities[task] = {}
            for modality in self.modalities:
                self.k_best_modalities[task][modality.modality_id] = (
                    self.unimodal_optimizer.get_k_best_results(modality, self.k, task)
                )


class MultimodalResults:
    def __init__(self, modality, task):
        self.modality_id = modality.modality_id
        self.task = task

        self.results = []


@dataclasses.dataclass
class MultimodalResultEntry:
    val_score: float
    modality_ids: list
    representations: list
    fusion_method: str
    representation_params: list
    train_score: float
    fusion_params: list
