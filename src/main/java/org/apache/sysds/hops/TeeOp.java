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

package org.apache.sysds.hops;

import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.Tee;
import org.apache.sysds.runtime.meta.DataCharacteristics;

import java.util.ArrayList;


public class TeeOp extends Hop {

	private final ArrayList<Hop> _outputs = new ArrayList<>();

	private TeeOp() {
		// default constructor
	}

	/**
	 * Takes in a single Hop input and gives two outputs
	 *
	 * @param input
	 * @param outputs
	 */
	public TeeOp(Hop input, ArrayList<Hop> outputs) {
		super(input.getName(), input.getDataType(), input._valueType);

		// add single input for this hop
		getInput().add(0, input);
		input.getParent().add(this);

		// output variables list to feed tee output into
		for (Hop out:  outputs) {
			_outputs.add(out);
		}

		// This characteristics are same as the input
		refreshSizeInformation();
	}

	@Override
	public boolean allowsAllExecTypes() {
		return false;
	}

	/**
	 * Computes the output matrix characteristics (rows, cols, nnz) based on worst-case output
	 * and/or input estimates. Should return null if dimensions are unknown.
	 *
	 * @param memo memory table
	 * @return output characteristics
	 */
	@Override
	protected DataCharacteristics inferOutputCharacteristics(MemoTable memo) {
		return null;
	}

	@Override
	public Lop constructLops() {
		// return already created Lops
		if (getLops() != null) {
			return getLops();
		}

		Tee teeLop = new Tee(getInput().get(0).constructLops(),
				getDataType(), getValueType());
		setOutputDimensions(teeLop);
		setLineNumbers(teeLop);
		setLops(teeLop);

		return getLops();
	}

	@Override
	protected ExecType optFindExecType(boolean transitive) {
		return ExecType.OOC;
	}

	@Override
	public String getOpString() {
		return "tee";
	}

	/**
	 * In memory-based optimizer mode (see OptimizerUtils.isMemoryBasedOptLevel()),
	 * the exectype is determined by checking this method as well as memory budget of this Hop.
	 * Please see findExecTypeByMemEstimate for more detail.
	 * <p>
	 * This method is necessary because not all operator are supported efficiently
	 * on GPU (for example: operations on frames and scalar as well as operations such as table).
	 *
	 * @return true if the Hop is eligible for GPU Exectype.
	 */
	@Override
	public boolean isGPUEnabled() {
		return false;
	}

	/**
	 * Computes the hop-specific output memory estimate in bytes. Should be 0 if not
	 * applicable.
	 *
	 * @param dim1 dimension 1
	 * @param dim2 dimension 2
	 * @param nnz  number of non-zeros
	 * @return memory estimate
	 */
	@Override
	protected double computeOutputMemEstimate(long dim1, long dim2, long nnz) {
		return 0;
	}

	/**
	 * Computes the hop-specific intermediate memory estimate in bytes. Should be 0 if not
	 * applicable.
	 *
	 * @param dim1 dimension 1
	 * @param dim2 dimension 2
	 * @param nnz  number of non-zeros
	 * @return memory estimate
	 */
	@Override
	protected double computeIntermediateMemEstimate(long dim1, long dim2, long nnz) {
		return 0;
	}

	/**
	 * Update the output size information for this hop.
	 */
	@Override
	public void refreshSizeInformation() {
		Hop input1 =  getInput().get(0);
		setDim1(input1.getDim1());
		setDim2(input1.getDim2());
		setNnz(input1.getNnz());
	}

	@Override
	public Object clone() throws CloneNotSupportedException {
		return null;
	}

	@Override
	public boolean compare(Hop that) {
		return false;
	}

	public Hop getOutput(int index) {
		return _outputs.get(index);
	}
}
