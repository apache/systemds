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

package org.apache.sysds.lops;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.instructions.InstructionUtils;


/**
 * Lop to perform reblock operation
 */
public class ReBlock extends Lop {
	public static final String OPCODE = Opcodes.RBLK.toString();
	
	private boolean _outputEmptyBlocks = true;
	
	private int _blocksize;

	public ReBlock(Lop input, int blen, DataType dt, ValueType vt, boolean outputEmptyBlocks, ExecType et)
	{
		super(Lop.Type.ReBlock, dt, vt);
		addInput(input);
		input.addOutput(this);
		
		_blocksize = blen;
		_outputEmptyBlocks = outputEmptyBlocks;
		
		if(et == ExecType.SPARK || et == ExecType.OOC) 
			lps.setProperties(inputs, et);
		else 
			throw new LopsException("Incorrect execution type for Reblock:" + et);
	}

	@Override
	public String toString() {
		return "Reblock - blocksize = " + _blocksize;
	}

	@Override
	public String getInstructions(String input1, String output) {
		return InstructionUtils.concatOperands(
			getExecType().name(),
			"rblk",
			getInputs().get(0).prepInputOperand(input1),
			prepOutputOperand(output),
			String.valueOf(_blocksize),
			String.valueOf(_outputEmptyBlocks));
	}
	
	// This function is replicated in Dag.java
	@SuppressWarnings("unused")
	private FileFormat getChildFormat(Lop node) {
		
		if(node.getOutputParameters().getFile_name() != null
				|| node.getOutputParameters().getLabel() != null)
		{
			return node.getOutputParameters().getFormat();
		}
		else
		{
			// Reblock lop should always have a single child
			if(node.getInputs().size() > 1)
				throw new LopsException(this.printErrorLocation() + "Should only have one child! \n");
			
			/*
			 * Return the format of the child node (i.e., input lop)
			 * No need of recursion here.. because
			 * 1) Reblock lop's input can either be DataLop or some intermediate computation
			 *    If it is Data then we just take its format (TEXT or BINARY)
			 *    If it is intermediate lop then it is always BINARY 
			 *      since we assume that all intermediate computations will be in Binary format
			 * 2) Note that Reblock job will never have any instructions in the mapper 
			 *    => the input lop (if it is other than Data) is always executed in a different job
			 */
			// return getChildFormat(node.getInputs().get(0));
			return node.getInputs().get(0).getOutputParameters().getFormat();
		}
	}
}
