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
package org.apache.sysml.runtime.instructions.gpu;

import java.util.HashMap;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.utils.GPUStatistics;

import jcuda.Pointer;

public class GPUDenseInputPointerFetcher implements java.lang.AutoCloseable {
	ExecutionContext _ec; GPUContext _gCtx; String _instName;
	HashMap<String, CPOperand> _inputMatrices = new HashMap<>();
	HashMap<String, MatrixObject> _inputMatrixObjects = new HashMap<>();
	HashMap<String, CPOperand> _inputScalars = new HashMap<>();
	CPOperand _output;
	public GPUDenseInputPointerFetcher(ExecutionContext ec, GPUContext gCtx, String instName, CPOperand output) {
		_ec = ec;
		_gCtx = gCtx;
		_instName = instName;
		_output = output;
	}
	public GPUDenseInputPointerFetcher add(String var, CPOperand in) {
		_inputMatrices.put(var, in);
		return this;
	}
	public GPUDenseInputPointerFetcher addScalar(String var, CPOperand in) {
		_inputScalars.put(var, in);
		return this;
	}
	public double getDouble(String var) {
		CPOperand in = _inputScalars.get(var);
		return _ec.getScalarInput(in.getName(), in.getValueType(), in.isLiteral()).getDoubleValue();
	}
	public long getLong(String var) {
		CPOperand in = _inputScalars.get(var);
		return _ec.getScalarInput(in.getName(), in.getValueType(), in.isLiteral()).getLongValue();
	}
	public int getInteger(String var) {
		CPOperand in = _inputScalars.get(var);
		return LibMatrixCUDA.toInt(_ec.getScalarInput(in.getName(), in.getValueType(), in.isLiteral()).getLongValue());
	}
	public Pointer getInputPointer(String var) {
		return LibMatrixCUDA.getDensePointer(_gCtx, getInputMatrixObject(var), _instName);
	}
	public long getInputNumRows(String var) {
		return getInputMatrixObject(var).getNumRows();
	}
	public long getInputNumColumns(String var) {
		return getInputMatrixObject(var).getNumColumns();
	}
	public MatrixObject getOutputMatrixObject(long numRows, long numCols) {
		boolean isFinegrainedStats = ConfigurationManager.isFinegrainedStatistics();
		long t0 = isFinegrainedStats ? System.nanoTime() : 0;
		Pair<MatrixObject, Boolean> mb = _ec.getDenseMatrixOutputForGPUInstruction(_output.getName(), numRows, numCols);
		if (isFinegrainedStats && mb.getValue()) GPUStatistics.maintainCPMiscTimes(_instName,
				GPUInstruction.MISC_TIMER_ALLOCATE_DENSE_OUTPUT, System.nanoTime() - t0);
		return  mb.getKey();
	}
	public Pointer getOutputPointer(long numRows, long numCols) {
		return LibMatrixCUDA.getDensePointer(_gCtx, getOutputMatrixObject(numRows, numCols), _instName);
	}
	public MatrixObject getInputMatrixObject(String var) {
		CPOperand in = _inputMatrices.get(var);
		if(!_inputMatrixObjects.containsKey(var)) {
			_inputMatrixObjects.put(var, _ec.getMatrixInputForGPUInstruction(in.getName(), _instName));
		}
		return _inputMatrixObjects.get(var);
	}
	public void validateDimensions(String var, long numRows, long numCols) {
		MatrixObject mo = getInputMatrixObject(var);
		if(numRows > 0 && mo.getNumRows() != numRows) {
			throw new DMLRuntimeException("Expected number of rows of subgrp_means to be " + numRows + ", but found " + mo.getNumRows());
		}
		else if(numCols > 0 && mo.getNumColumns() != numCols) {
			throw new DMLRuntimeException("Expected number of columns of subgrp_means to be " + numCols + ", but found " + mo.getNumColumns());
		}
	}
	@Override
	public void close() {
		for(CPOperand in : _inputMatrices.values()) {
			_ec.releaseMatrixInputForGPUInstruction(in.getName());
		}
		_ec.releaseMatrixOutputForGPUInstruction(_output.getName());
	}
}