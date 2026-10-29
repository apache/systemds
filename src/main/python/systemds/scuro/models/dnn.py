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
from systemds.scuro.models.model import Model


class DNN(Model):
    """Base class for the neural task models, carrying their training budget.

    The budget is a number of optimizer steps -- mini-batches -- and not a
    number of epochs. An epoch is not a comparable unit of work: one epoch is
    two mini-batches on a StressID fold and over two hundred on an MM-IMDb
    fold, so an epoch count says nothing on its own about how much optimisation
    a model got, and a budget expressed in epochs cannot be compared across
    datasets at all. A step is the same unit everywhere (at a fixed batch
    size), so `num_steps` means the same thing on every dataset, and a budget
    smaller than one pass over the data is expressible rather than rounded up.

    `num_steps` is None only for the older models that still run their own
    epoch loop and do not use `train_batches`.
    """

    def __init__(self, name, num_steps=None):
        super().__init__(name)
        self.num_steps = num_steps

    def drop_last_batch(self, n_samples, batch_size):
        """Whether the loader should drop an uneven final batch.

        Every step has to be the same amount of work, or the budget stops being
        a unit of work at all; a short tail batch also breaks BatchNorm when it
        holds a single sample. Dropping it is only safe while at least one full
        batch remains, which is not a given on the small folds (StressID's are
        a handful of batches wide).
        """
        return int(n_samples) >= int(batch_size) * 2

    def train_batches(self, loader, stats=None):
        """Yield mini-batches until the step budget is spent.

        The loader is re-iterated as often as the budget needs, so a budget of
        more than one pass reshuffles between passes exactly as an epoch loop
        would, and a budget of less than one pass simply stops part-way through
        the first one.
        """
        remaining = max(0, int(self.num_steps or 0))
        if not len(loader):
            return
        while remaining > 0:
            if stats is not None:
                stats.count_pass()
            for batch in loader:
                yield batch
                remaining -= 1
                if remaining == 0:
                    return
