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

package org.apache.sysml.runtime.instructions.flink.functions;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.util.Collector;
import org.apache.sysml.runtime.matrix.data.BinaryBlockToTextCellConverter;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;

public class ConvertMatrixBlockToIJVLines implements FlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, String> {

	private final int brlen;
	private final int bclen;

	public ConvertMatrixBlockToIJVLines(int brlen, int bclen) {
		this.brlen = brlen;
		this.bclen = bclen;
	}

	@Override
	public void flatMap(Tuple2<MatrixIndexes, MatrixBlock> kv, Collector<String> out) {
		final BinaryBlockToTextCellConverter converter = new BinaryBlockToTextCellConverter();
		converter.setBlockSize(brlen, bclen);
		converter.convert(kv.f0, kv.f1);

		while (converter.hasNext()) {
			out.collect(converter.next().getValue().toString());
		}
	}
}
