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

package org.apache.sysml.runtime.instructions.cp;

import java.io.IOException;

import org.apache.sysml.lops.PickByCount.OperationTypes;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.MetaData;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.NumItemsByEachReducerMetaData;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.runtime.util.UtilFunctions;

public class QuantilePickCPInstruction extends BinaryCPInstruction
{
	
	private OperationTypes _type = null;
	private boolean _inmem = true;
	
	public QuantilePickCPInstruction(Operator op, CPOperand in, CPOperand out, OperationTypes type, boolean inmem, String opcode, String istr){
		this(op, in, null, out, type, inmem, opcode, istr);
	}
	
	public QuantilePickCPInstruction(Operator op, CPOperand in, CPOperand in2, CPOperand out,  OperationTypes type, boolean inmem, String opcode, String istr){
		super(op, in, in2, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.QPick;
		
		_type = type;
		_inmem = inmem;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static QuantilePickCPInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		//sanity check opcode
		if ( !opcode.equalsIgnoreCase("qpick") ) {
			throw new DMLRuntimeException("Unknown opcode while parsing a QuantilePickCPInstruction: " + str);
		}
		
		//instruction parsing
		if( parts.length == 4 )
		{
			//instructions of length 4 originate from unary - mr-iqm
			//TODO this should be refactored to use pickvaluecount lops
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			OperationTypes ptype = OperationTypes.IQM;
			boolean inmem = false;
			return new QuantilePickCPInstruction(null, in1, in2, out, ptype, inmem, opcode, str);			
		}
		else if( parts.length == 5 )
		{
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand out = new CPOperand(parts[2]);
			OperationTypes ptype = OperationTypes.valueOf(parts[3]);
			boolean inmem = Boolean.parseBoolean(parts[4]);
			return new QuantilePickCPInstruction(null, in1, out, ptype, inmem, opcode, str);
		}
		else if( parts.length == 6 )
		{
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			OperationTypes ptype = OperationTypes.valueOf(parts[4]);
			boolean inmem = Boolean.parseBoolean(parts[5]);
			return new QuantilePickCPInstruction(null, in1, in2, out, ptype, inmem, opcode, str);
		}
		
		return null;
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLRuntimeException 
	{
		switch( _type ) 
		{
			case VALUEPICK: 
				if( _inmem ) //INMEM VALUEPICK
				{
					MatrixBlock matBlock = ec.getMatrixInput(input1.getName());

					if ( input2.getDataType() == DataType.SCALAR ) {
						ScalarObject quantile = ec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral());
						double picked = matBlock.pickValue(quantile.getDoubleValue());
						ec.setScalarOutput(output.getName(), new DoubleObject(picked));
					} 
					else {
						MatrixBlock quantiles = ec.getMatrixInput(input2.getName());
						MatrixBlock resultBlock = (MatrixBlock) matBlock.pickValues(quantiles, new MatrixBlock());
						quantiles = null;
						ec.releaseMatrixInput(input2.getName());
						ec.setMatrixOutput(output.getName(), resultBlock);
					}
					ec.releaseMatrixInput(input1.getName());										
				}
				else //MR VALUEPICK
				{
					MatrixObject mat = ec.getMatrixObject(input1.getName());
					String fname = mat.getFileName();
					MetaData mdata = mat.getMetaData();
					ScalarObject pickindex = ec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral());
					
					if ( mdata != null ) {
						try {
							double picked = MapReduceTool.pickValue(fname, (NumItemsByEachReducerMetaData) mdata, pickindex.getDoubleValue());
							ec.setVariable(output.getName(), new DoubleObject(picked));
						} catch (Exception e ) {
							throw new DMLRuntimeException(e);
						}
					}
					else {
						throw new DMLRuntimeException("Unexpected error while executing ValuePickCP: otherMetaData for file (" + fname + ") not found." );
					}
				}				
				break;

			case MEDIAN:
				if( _inmem ) //INMEM MEDIAN
				{
					double picked = ec.getMatrixInput(input1.getName()).median();
					ec.setScalarOutput(output.getName(), new DoubleObject(picked));
					ec.releaseMatrixInput(input1.getName());
					break;
				}
				else //MR MEDIAN
				{
					MatrixObject mat1 = (MatrixObject)ec.getVariable(input1.getName());
					String fname1 = mat1.getFileName();
					MetaData mdata1 = mat1.getMetaData();
					
					if ( mdata1 != null ) {
						try {
							double median = MapReduceTool.median(fname1, (NumItemsByEachReducerMetaData) mdata1);
							ec.setVariable(output.getName(), new DoubleObject(median));
						} catch (Exception e ) {
							throw new DMLRuntimeException(e);
						}
					}
					else {
						throw new DMLRuntimeException("Unexpected error while executing ValuePickCP: otherMetaData for file (" + fname1 + ") not found." );
					}
				}
				break;
				
			case IQM:
				if( _inmem ) //INMEM IQM
				{
					MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
					double iqm = matBlock1.interQuartileMean();
					ec.releaseMatrixInput(input1.getName());
					ec.setScalarOutput(output.getName(), new DoubleObject(iqm));
				}
				else //MR IQM
				{
					MatrixObject inputMatrix = (MatrixObject)ec.getVariable(input1.getName());
					ScalarObject iqsum = ec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral());
					
					double[] q25 = null;
					double[] q75 = null;
					try {
						q25 = MapReduceTool.pickValueWeight(inputMatrix.getFileName(), (NumItemsByEachReducerMetaData) inputMatrix.getMetaData(), 0.25, false);
						q75 = MapReduceTool.pickValueWeight(inputMatrix.getFileName(), (NumItemsByEachReducerMetaData) inputMatrix.getMetaData(), 0.75, false);
					} catch (IOException e1) {
						throw new DMLRuntimeException(e1);
					}
					
					double sumwt = UtilFunctions.getTotalLength((NumItemsByEachReducerMetaData) ec.getMetaData(input1.getName()));
					double q25d = sumwt*0.25;
					double q75d = sumwt*0.75;
					
					// iqsum = interQuartileSum that includes complete portions of q25 and q75
					//   . exclude top portion of q25 and bottom portion of q75 
					double q25entry_weight = q25[0]*q25[1];
					double q25portion_include = (q25[2]-q25d)*q25[0];
					double q25portion_exclude = q25entry_weight-q25portion_include;
					double q75portion_exclude = (q75[2]-q75d)*q75[0];
					
					double mriqm = (iqsum.getDoubleValue() - q25portion_exclude - q75portion_exclude)/(sumwt*0.5);

					ec.setScalarOutput(output.getName(), new DoubleObject(mriqm));
				}
				break;
				
			default:
				throw new DMLRuntimeException("Unsupported qpick operation type: "+_type);
		}
	}
}
