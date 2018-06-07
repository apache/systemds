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

package org.apache.sysml.runtime.controlprogram.paramserv;

import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import org.apache.sysml.hops.Hop;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.FrameObject;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.CtableCPInstruction;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.DataGenCPInstruction;
import org.apache.sysml.runtime.instructions.cp.ListObject;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MetaDataFormat;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.DataConverter;

public class ParamservUtils {

	/**
	 * Deep copy the list object
	 *
	 * @param lo list object
	 * @return a new copied list object
	 */
	public static ListObject copyList(ListObject lo) {
		if (lo.getLength() == 0) {
			return lo;
		}
		List<Data> newData = IntStream.range(0, lo.getLength()).mapToObj(i -> {
			Data oldData = lo.slice(i);
			if (oldData instanceof MatrixObject) {
				MatrixObject mo = (MatrixObject) oldData;
				return sliceMatrix(mo, 1, mo.getNumRows());
			} else if (oldData instanceof ListObject || oldData instanceof FrameObject) {
				throw new DMLRuntimeException("Copy list: does not support list or frame.");
			} else {
				return oldData;
			}
		}).collect(Collectors.toList());
		return new ListObject(newData, lo.getNames());
	}

	public static void cleanupListObject(ExecutionContext ec, ListObject lo) {
		ec.getVariables().removeAllIn(new HashSet<>(lo.getNames()));
		lo.getData().forEach(ParamservUtils::cleanupData);
	}

	public static void cleanupData(Data data) {
		if (!(data instanceof CacheableData))
			return;
		CacheableData<?> cd = (CacheableData<?>) data;
		cd.enableCleanup(true);
		cd.clearData();
	}

	public static MatrixObject newMatrixObject() {
		return new MatrixObject(Expression.ValueType.DOUBLE, null,
				new MetaDataFormat(new MatrixCharacteristics(-1, -1, -1, -1), OutputInfo.BinaryBlockOutputInfo,
						InputInfo.BinaryBlockInputInfo));
	}

	/**
	 * Slice the matrix
	 *
	 * @param mo input matrix
	 * @param rl low boundary
	 * @param rh high boundary
	 * @return new sliced matrix
	 */
	public static MatrixObject sliceMatrix(MatrixObject mo, long rl, long rh) {
		MatrixObject result = newMatrixObject();
		MatrixBlock tmp = mo.acquireRead();
		result.acquireModify(tmp.slice((int) rl - 1, (int) rh - 1, 0, tmp.getNumColumns() - 1, new MatrixBlock()));
		mo.release();
		result.release();
		result.enableCleanup(false);
		return result;
	}

	public static MatrixObject generatePermutation(MatrixObject mo, ExecutionContext ec) {
		// Create the sequence
		double[] data = LongStream.range(1, mo.getNumRows() + 1).mapToDouble(l -> l).toArray();
		MatrixBlock seqMB = DataConverter.convertToMatrixBlock(data, true);
		MatrixObject seq = ParamservUtils.newMatrixObject();
		seq.acquireModify(seqMB);
		seq.release();
		ec.setVariable("seq", seq);

		// Generate a sample
		DataGenCPInstruction sampleInst = new DataGenCPInstruction(null, Hop.DataGenMethod.SAMPLE, null,
				new CPOperand("sample", Expression.ValueType.DOUBLE, Expression.DataType.MATRIX),
				new CPOperand(String.valueOf(mo.getNumRows()), Expression.ValueType.INT, Expression.DataType.SCALAR,
						true), new CPOperand("1", Expression.ValueType.INT, Expression.DataType.SCALAR, true),
				(int) mo.getNumRowsPerBlock(), (int) mo.getNumColumnsPerBlock(), mo.getNumRows(), false, -1,
				Hop.DataGenMethod.SAMPLE.name().toLowerCase(), "sample");
		ec.setVariable("sample", ParamservUtils.newMatrixObject());
		sampleInst.processInstruction(ec);

		// Combine the sequence and sample as a table
		CtableCPInstruction tableInst = new CtableCPInstruction(
				new CPOperand("seq", Expression.ValueType.DOUBLE, Expression.DataType.MATRIX),
				new CPOperand("sample", Expression.ValueType.DOUBLE, Expression.DataType.MATRIX),
				new CPOperand("1.0", Expression.ValueType.DOUBLE, Expression.DataType.SCALAR, true),
				new CPOperand("permutation", Expression.ValueType.DOUBLE, Expression.DataType.MATRIX), "-1", true, "-1",
				true, true, false, "ctableexpand", "table");
		MatrixObject permutation = ParamservUtils.newMatrixObject();
		ec.setVariable("permutation", permutation);
		tableInst.processInstruction(ec);
		return permutation;
	}
}
