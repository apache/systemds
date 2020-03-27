/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.instructions.spark;

import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.functionobjects.IndexFunction;
import org.tugraz.sysds.runtime.functionobjects.ReduceAll;
import org.tugraz.sysds.runtime.functionobjects.ReduceCol;
import org.tugraz.sysds.runtime.functionobjects.ReduceRow;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.lineage.LineageItem;
import org.tugraz.sysds.runtime.lineage.LineageItemUtils;
import org.tugraz.sysds.runtime.lineage.LineageTraceable;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;

public abstract class ComputationSPInstruction extends SPInstruction implements LineageTraceable {
	public CPOperand output;
	public CPOperand input1, input2, input3;

	protected ComputationSPInstruction(SPType type, Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(type, op, opcode, istr);
		input1 = in1;
		input2 = in2;
		input3 = null;
		output = out;
	}

	protected ComputationSPInstruction(SPType type, Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String opcode, String istr) {
		super(type, op, opcode, istr);
		input1 = in1;
		input2 = in2;
		input3 = in3;
		output = out;
	}

	public String getOutputVariableName() {
		return output.getName();
	}

	protected void updateUnaryOutputDataCharacteristics(SparkExecutionContext sec) {
		updateUnaryOutputDataCharacteristics(sec, input1.getName(), output.getName());
	}

	protected void updateUnaryOutputDataCharacteristics(SparkExecutionContext sec, String nameIn, String nameOut) {
		DataCharacteristics dc1 = sec.getDataCharacteristics(nameIn);
		DataCharacteristics dcOut = sec.getDataCharacteristics(nameOut);
		if(!dcOut.dimsKnown()) {
			if(!dc1.dimsKnown())
				throw new DMLRuntimeException("The output dimensions are not specified and cannot be inferred from input:" + dc1.toString() + " " + dcOut.toString());
			else
				dcOut.set(dc1.getRows(), dc1.getCols(), dc1.getBlocksize(), dc1.getBlocksize());
		}
	}

	protected void updateBinaryOutputDataCharacteristics(SparkExecutionContext sec) {
		DataCharacteristics dcIn1 = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics dcIn2 = sec.getDataCharacteristics(input2.getName());
		DataCharacteristics dcOut = sec.getDataCharacteristics(output.getName());
		boolean outer = (dcIn1.getRows()>1 && dcIn1.getCols()==1 && dcIn2.getRows()==1 && dcIn2.getCols()>1);
		
		if(!dcOut.dimsKnown()) {
			if(!dcIn1.dimsKnown())
				throw new DMLRuntimeException("The output dimensions are not specified and cannot be inferred from input:" + dcIn1.toString() + " " + dcIn2.toString() + " " + dcOut.toString());
			else if(outer)
				sec.getDataCharacteristics(output.getName()).set(dcIn1.getRows(), dcIn2.getCols(), dcIn1.getBlocksize(), dcIn2.getBlocksize());
			else
				sec.getDataCharacteristics(output.getName()).set(dcIn1.getRows(), dcIn1.getCols(), dcIn1.getBlocksize(), dcIn1.getBlocksize());
		}
	}

	protected void updateBinaryTensorOutputDataCharacteristics(SparkExecutionContext sec) {
		DataCharacteristics dcIn1 = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics dcIn2 = sec.getDataCharacteristics(input2.getName());
		DataCharacteristics dcOut = sec.getDataCharacteristics(output.getName());

		// TODO the dcOut dims will not be accurate here, because set output dimensions currently do only support
		//  matrix size informations. Changing this requires changes in `Hop` and `OutputParameters`.
		if(!dcOut.dimsKnown()) {
			if(!dcIn1.dimsKnown())
				throw new DMLRuntimeException("The output dimensions are not specified and cannot be inferred from input:" + dcIn1.toString() + " " + dcIn2.toString() + " " + dcOut.toString());
			else
				dcOut.set(dcIn1);
		}
		// TODO remove this once dcOut dims are accurate if known
		dcOut.set(dcIn1);
	}

	protected void updateUnaryAggOutputDataCharacteristics(SparkExecutionContext sec, IndexFunction ixFn) {
		DataCharacteristics mc1 = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
		if( mcOut.dimsKnown() )
			return;
		
		if(!mc1.dimsKnown()) {
			throw new DMLRuntimeException("The output dimensions are not specified and "
				+ "cannot be inferred from input:" + mc1.toString() + " " + mcOut.toString());
		}
		else {
			//infer statistics from input based on operator
			if( ixFn instanceof ReduceAll )
				mcOut.set(1, 1, mc1.getBlocksize(), mc1.getBlocksize());
			else if( ixFn instanceof ReduceCol )
				mcOut.set(mc1.getRows(), 1, mc1.getBlocksize(), mc1.getBlocksize());
			else if( ixFn instanceof ReduceRow )
				mcOut.set(1, mc1.getCols(), mc1.getBlocksize(), mc1.getBlocksize());
		}
	}
	
	@Override
	public LineageItem[] getLineageItems(ExecutionContext ec) {
		return new LineageItem[]{new LineageItem(output.getName(), getOpcode(),
			LineageItemUtils.getLineage(ec, input1, input2, input3))};
	}
}
