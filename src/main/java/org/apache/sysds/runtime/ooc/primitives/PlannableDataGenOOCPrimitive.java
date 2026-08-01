/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.ooc.primitives;

import java.util.List;
import java.util.function.Function;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.ooc.memory.ReservationBudget;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.ooc.stream.AllocatedOOCStream;
import org.apache.sysds.runtime.ooc.stream.StreamContext;
import org.apache.sysds.runtime.ooc.util.OOCInstructionUtils;
import org.apache.sysds.runtime.ooc.util.OOCUtils;

public class PlannableDataGenOOCPrimitive extends OOCPrimitive {
	private final OOCStreamable<IndexedMatrixValue> _output;
	private final Function<MatrixIndexes, MatrixBlock> _operation;

	public PlannableDataGenOOCPrimitive(OOCStreamable<IndexedMatrixValue> output,
		Function<MatrixIndexes, MatrixBlock> operation, StreamContext context) {
		super(context, List.of());
		_output = output;
		_operation = operation;
	}

	@Override
	protected void inferPatternsInternal() {
		if(_pattern.isUnset())
			_pattern = OOCAccessPattern.ANY;
		inferParentPatterns();
	}

	@Override
	protected void requestPatternInternal(OOCAccessPattern accessPattern) {
		_pattern = accessPattern;
	}

	@Override
	protected void startExecution() {
		OOCStream<MatrixIndexes> work = new SubscribableTaskQueue<>();
		OOCStream<IndexedMatrixValue> output = _output.getWriteStream();
		long outputBytes = OOCUtils.estimateOutputTileBytes(_output.getDataCharacteristics());
		AllocatedOOCStream<MatrixIndexes> admitted = new AllocatedOOCStream<>(work, _allowance, ignored -> outputBytes);
		getContext().addOutStream(output);
		OOCInstructionUtils.submitOOCTasks(admitted, callback -> {
			ReservationBudget budget = AllocatedOOCStream.detachBudget(callback);
			try {
				MatrixIndexes indexes = callback.get();
				OOCUtils.enqueueExact(output, new IndexedMatrixValue(indexes, _operation.apply(indexes)), budget);
				budget = null;
			}
			finally {
				if(budget != null)
					budget.close();
			}
		}, getContext()).thenRun(output::closeInput).exceptionally(error -> {
			output.propagateFailure(DMLRuntimeException.of(error));
			return null;
		}).thenRun(this::onComplete);

		OOCInstructionUtils.submitOOCTask(() -> {
			for(MatrixIndexes indexes : OOCUtils.getAccessPattern(_output.getDataCharacteristics(), _pattern))
				work.enqueue(indexes);
			work.closeInput();
		}, new StreamContext().addOutStream(work));
	}
}
