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

import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.ooc.stream.StreamContext;
import org.apache.sysds.runtime.ooc.util.OOCInstructionUtils;

public class MappingOOCPrimitive extends OOCPrimitive {
	private final OOCStreamable<IndexedMatrixValue> _input;
	private final OOCStreamable<IndexedMatrixValue> _output;
	private final Function<IndexedMatrixValue, MatrixBlock> _operation;

	public MappingOOCPrimitive(OOCStreamable<IndexedMatrixValue> input, OOCStreamable<IndexedMatrixValue> output,
		Function<IndexedMatrixValue, MatrixBlock> operation, StreamContext context) {
		this(input.getPrimitive(), input, output, operation, context);
	}

	private MappingOOCPrimitive(OOCPrimitive inputPrimitive, OOCStreamable<IndexedMatrixValue> input,
		OOCStreamable<IndexedMatrixValue> output, Function<IndexedMatrixValue, MatrixBlock> operation,
		StreamContext context) {
		super(context, inputPrimitive == null ? List.of() : List.of(inputPrimitive));
		_input = input;
		_output = output;
		_operation = operation;
	}

	@Override
	protected void inferPatternsInternal() {
		OOCAccessPattern inputPattern = getChildren().isEmpty() ? OOCAccessPattern.ANY : getChildren().get(0)
			.getAccessPattern();
		_pattern = _pattern.preferred(inputPattern);
		inferParentPatterns();
	}

	@Override
	protected void requestPatternInternal(OOCAccessPattern accessPattern) {
		_pattern = _pattern.preferred(accessPattern);
		if(!getChildren().isEmpty())
			getChildren().get(0).requestPattern(accessPattern);
	}

	@Override
	protected void startExecution() {
		OOCStream<IndexedMatrixValue> input = _input.getReadStream();
		OOCStream<IndexedMatrixValue> output = _output.getWriteStream();
		OOCInstructionUtils
			.submitAdmittedOOCTasks(input, output,
				value -> new IndexedMatrixValue(value.getIndexes(), _operation.apply(value)), _allowance, getContext())
			.thenRun(this::onComplete);
	}
}
