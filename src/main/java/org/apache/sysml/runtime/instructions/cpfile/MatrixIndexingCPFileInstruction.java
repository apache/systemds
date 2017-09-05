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

package org.apache.sysml.runtime.instructions.cpfile;

import org.apache.sysml.lops.LeftIndex;
import org.apache.sysml.lops.RightIndex;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.IndexingCPInstruction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.SimpleOperator;
import org.apache.sysml.runtime.util.IndexRange;
import org.apache.sysml.runtime.util.MapReduceTool;

/**
 * This instruction is used if a single partition is too large to fit in memory.
 * Hence, the partition is not read but we just return a new matrix with the
 * respective partition file name. For this reason this is a no-op but due to
 * the requirement for direct partition access only applicable for ROWWISE and
 * COLWISE partition formats. 
 * 
 */
public final class MatrixIndexingCPFileInstruction extends IndexingCPInstruction {

	private MatrixIndexingCPFileInstruction(Operator op, CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl,
			CPOperand cu, CPOperand out, String opcode, String istr) {
		super(op, in, rl, ru, cl, cu, out, opcode, istr);
	}

	public static MatrixIndexingCPFileInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase(RightIndex.OPCODE) ) {
			if ( parts.length == 7 ) {
				CPOperand in, rl, ru, cl, cu, out;
				in = new CPOperand(parts[1]);
				rl = new CPOperand(parts[2]);
				ru = new CPOperand(parts[3]);
				cl = new CPOperand(parts[4]);
				cu = new CPOperand(parts[5]);
				out = new CPOperand(parts[6]);
				return new MatrixIndexingCPFileInstruction(new SimpleOperator(null), in, rl, ru, cl, cu, out, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else if ( parts[0].equalsIgnoreCase(LeftIndex.OPCODE)) 
		{
			throw new DMLRuntimeException("Invalid opcode while parsing a MatrixIndexingCPFileInstruction: " + str);	
		}
		else 
		{
			throw new DMLRuntimeException("Unknown opcode while parsing a MatrixIndexingCPFileInstruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException 
	{	
		String opcode = getOpcode();
		IndexRange ixrange = getIndexRange(ec).add(1);
		MatrixObject mo = ec.getMatrixObject(input1.getName());
		
		if( mo.isPartitioned() && opcode.equalsIgnoreCase(RightIndex.OPCODE) ) 
		{
			MatrixFormatMetaData meta = (MatrixFormatMetaData)mo.getMetaData();
			MatrixCharacteristics mc = meta.getMatrixCharacteristics();
			String pfname = mo.getPartitionFileName( ixrange, mc.getRowsPerBlock(), mc.getColsPerBlock());
			
			if( MapReduceTool.existsFileOnHDFS(pfname) )
			{
				MatrixObject out = ec.getMatrixObject(output.getName());
				
				//create output matrix object				
				MatrixObject mobj = new MatrixObject(mo.getValueType(), pfname );
				mobj.setDataType( DataType.MATRIX );
				mobj.setVarName( out.getVarName() );
				MatrixCharacteristics mcNew = null;
				switch( mo.getPartitionFormat() )
				{
					case ROW_WISE:
						mcNew = new MatrixCharacteristics( 1, mc.getCols(), mc.getRowsPerBlock(), mc.getColsPerBlock() );
						break;
					case ROW_BLOCK_WISE_N:
						mcNew = new MatrixCharacteristics( mo.getPartitionSize(), mc.getCols(), mc.getRowsPerBlock(), mc.getColsPerBlock() );
						break;	
					case COLUMN_WISE:
						mcNew = new MatrixCharacteristics( mc.getRows(), 1, mc.getRowsPerBlock(), mc.getColsPerBlock() );
						break;
					case COLUMN_BLOCK_WISE_N:
						mcNew = new MatrixCharacteristics( mc.getRows(), mo.getPartitionSize(), mc.getRowsPerBlock(), mc.getColsPerBlock() );
						break;	
					default:
						throw new DMLRuntimeException("Unsupported partition format for CP_FILE "+RightIndex.OPCODE+": "+ mo.getPartitionFormat());
				}
				
				MatrixFormatMetaData metaNew = new MatrixFormatMetaData(mcNew,meta.getOutputInfo(),meta.getInputInfo());
				mobj.setMetaData(metaNew);	 
				
				//put output object into symbol table
				ec.setVariable(output.getName(), mobj);
			}
			else
			{
				//will return an empty matrix partition 
				MatrixBlock resultBlock = mo.readMatrixPartition( ixrange );
				ec.setMatrixOutput(output.getName(), resultBlock, getExtendedOpcode());
			}
		}
		else
		{
			throw new DMLRuntimeException("Invalid opcode or index predicate for MatrixIndexingCPFileInstruction: " + instString);	
		}
	}
}