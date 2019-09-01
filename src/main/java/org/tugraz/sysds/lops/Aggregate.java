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

package org.tugraz.sysds.lops;

import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.lops.PartialAggregate.CorrectionLocationType;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;


/**
 * Lop to represent an aggregation.
 * It is used in rowsum, colsum, etc. 
 */

public class Aggregate extends Lop 
{
	/** Aggregate operation types **/
	
	public enum OperationTypes {
		Sum, Product, SumProduct, Min, Max, Trace,
		KahanSum, KahanSumSq, KahanTrace, Mean, Var, MaxIndex, MinIndex
	}
	OperationTypes operation;
 
	public Aggregate(Lop input, Aggregate.OperationTypes op, DataType dt, ValueType vt, ExecType et ) {
		super(Lop.Type.Aggregate, dt, vt);
		init ( input, op, dt, vt, et );
	}
	
	private void init (Lop input, Aggregate.OperationTypes op, DataType dt, ValueType vt, ExecType et ) {
		operation = op;	
		this.addInput(input);
		input.addOutput(this);
		lps.setProperties(inputs, et);
	}
	
	// this function must be invoked during hop-to-lop translation
	public void setupCorrectionLocation(CorrectionLocationType loc) {

	}
	
	@Override
	public String toString() {
		return "Operation: " + operation;
	}

	/**
	 * method to get operation type
	 * 
	 * @return operator type
	 */
	public OperationTypes getOperationType() {
		return operation;
	}
	
	
	private String getOpcode() {
		switch(operation) {
		case Sum: 
		case Trace: 
			return "a+"; 
		case Mean: 
			return "amean";
		case Var:
			return "avar";
		case Product: 
			return "a*"; 
		case Min: 
			return "amin"; 
		case Max: 
			return "amax"; 
		case MaxIndex:
			return "arimax";
		case MinIndex:
			return "arimin";
		case KahanSum:
		case KahanTrace: 
			return "ak+";
		case KahanSumSq:
			return "asqk+";
		default:
			throw new UnsupportedOperationException(this.printErrorLocation() + "Instruction is not defined for Aggregate operation: " + operation);
		}
	}

	@Override
	public String getInstructions(String input1, String output) {
		return InstructionUtils.concatOperands(
			getExecType().name(),
			getOpcode(),
			getInputs().get(0).prepInputOperand(input1),
			prepOutputOperand(output));
	}
}
