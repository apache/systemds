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

package org.apache.sysml.lops;

import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;


public class LeftIndex extends Lop 
{
	
	/**
	 * Constructor to setup a LeftIndexing operation.
	 * Example: A[i:j, k:l] = B;
	 *      
	 * 
	 * @param input
	 * @param op
	 * @return 
	 * @throws LopsException
	 */
	
	private void init(Lop lhsMatrix, Lop rhsMatrix, Lop rowL, Lop rowU, Lop colL, Lop colU, ExecType et) throws LopsException {
		/*
		 * A[i:j, k:l] = B;
		 * B -> rhsMatrix
		 * A -> lhsMatrix
		 * i,j -> rowL, rowU
		 * k,l -> colL, colU
		 */
		this.addInput(lhsMatrix);
		this.addInput(rhsMatrix);
		this.addInput(rowL);
		this.addInput(rowU);
		this.addInput(colL);
		this.addInput(colU);
		
		lhsMatrix.addOutput(this);		
		rhsMatrix.addOutput(this);		
		rowL.addOutput(this);
		rowU.addOutput(this);
		colL.addOutput(this);
		colU.addOutput(this);

		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		if ( et == ExecType.MR ) {
			throw new LopsException(this.printErrorLocation() + "LeftIndexing lop is undefined for MR runtime");
		} 
		else {
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
	}
	
	public LeftIndex(
			Lop lhsInput, Lop rhsInput, Lop rowL, Lop rowU, Lop colL, Lop colU, DataType dt, ValueType vt, ExecType et)
			throws LopsException {
		super(Lop.Type.LeftIndex, dt, vt);
		init(lhsInput, rhsInput, rowL, rowU, colL, colU, et);
	}
	
	boolean isBroadcast = false;
	public LeftIndex(
			Lop lhsInput, Lop rhsInput, Lop rowL, Lop rowU, Lop colL, Lop colU, DataType dt, ValueType vt, ExecType et, boolean isBroadcast)
			throws LopsException {
		super(Lop.Type.LeftIndex, dt, vt);
		this.isBroadcast = isBroadcast;
		init(lhsInput, rhsInput, rowL, rowU, colL, colU, et);
	}

	private String getOpcode() {
		if(isBroadcast)
			return "mapLeftIndex";
		else
			return "leftIndex";
	}
	
	@Override
	public String getInstructions(String lhsInput, String rhsInput, String rowl, String rowu, String coll, String colu, String output) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepInputOperand(lhsInput));
		sb.append( OPERAND_DELIMITOR );
		
		if ( getInputs().get(1).getDataType() == DataType.SCALAR ) {
			sb.append( getInputs().get(1).prepScalarInputOperand(getExecType()));
		}
		else {
			sb.append( getInputs().get(1).prepInputOperand(rhsInput));
		}
		sb.append( OPERAND_DELIMITOR );
		
		// rowl, rowu
		sb.append(getInputs().get(2).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );	
		sb.append(getInputs().get(3).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );	
		
		// rowl, rowu
		sb.append(getInputs().get(4).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );	
		sb.append(getInputs().get(5).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );	
		
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}

	@Override
	public String toString() {
		return "leftIndex";
	}

}
