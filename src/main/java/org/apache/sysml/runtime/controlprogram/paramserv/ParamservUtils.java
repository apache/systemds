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

import org.apache.sysml.lops.RightIndex;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.FrameObject;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.ListObject;
import org.apache.sysml.runtime.instructions.cp.MatrixIndexingCPInstruction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MetaDataFormat;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.OutputInfo;

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
		List<Data> newData = lo.getNames().stream().map(name -> {
			Data oldData = lo.slice(name);
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
		if (data instanceof CacheableData) {
			CacheableData cd = (CacheableData) data;
			cd.enableCleanup(true);
			cd.clearData();
		}
	}

	/**
	 * Slice the matrix
	 * @param mo input matrix
	 * @param rl low boundary
	 * @param rh high boundary
	 * @return new sliced matrix
	 */
	public static MatrixObject sliceMatrix(MatrixObject mo, long rl, long rh) {
		ExecutionContext tmpEC = ExecutionContextFactory.createContext();
		MatrixObject result = new MatrixObject(Expression.ValueType.DOUBLE, null,
				new MetaDataFormat(new MatrixCharacteristics(-1, -1, -1, -1), OutputInfo.BinaryBlockOutputInfo,
						InputInfo.BinaryBlockInputInfo));
		tmpEC.setVariable("out", result);
		MatrixIndexingCPInstruction inst = new MatrixIndexingCPInstruction(new CPOperand("in"),
				new CPOperand(String.valueOf(rl), Expression.ValueType.INT, Expression.DataType.SCALAR, true),
				new CPOperand(String.valueOf(rh), Expression.ValueType.INT, Expression.DataType.SCALAR, true),
				new CPOperand("1", Expression.ValueType.INT, Expression.DataType.SCALAR, true),
				new CPOperand(String.valueOf(mo.getNumColumns()), Expression.ValueType.INT, Expression.DataType.SCALAR,
						true), new CPOperand("out"), RightIndex.OPCODE, "slice matrix");
		tmpEC.setVariable("in", mo);
		inst.processInstruction(tmpEC);
		return result;
	}

}
