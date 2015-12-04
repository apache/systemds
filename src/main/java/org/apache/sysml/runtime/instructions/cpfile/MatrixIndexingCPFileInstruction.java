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

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.MatrixIndexingCPInstruction;
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
public class MatrixIndexingCPFileInstruction extends MatrixIndexingCPInstruction 
{
	
	public MatrixIndexingCPFileInstruction(Operator op, CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr)
	{
		super( op, in, rl, ru, cl, cu, out, opcode, istr );
	}
	
	public MatrixIndexingCPFileInstruction(Operator op, CPOperand lhsInput, CPOperand rhsInput, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr)
	{
		super( op, lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, istr);
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase("rangeReIndex") ) {
			if ( parts.length == 7 ) {
				// Example: rangeReIndex:mVar1:Var2:Var3:Var4:Var5:mVar6
				CPOperand in, rl, ru, cl, cu, out;
				in = new CPOperand();
				rl = new CPOperand();
				ru = new CPOperand();
				cl = new CPOperand();
				cu = new CPOperand();
				out = new CPOperand();
				in.split(parts[1]);
				rl.split(parts[2]);
				ru.split(parts[3]);
				cl.split(parts[4]);
				cu.split(parts[5]);
				out.split(parts[6]);
				return new MatrixIndexingCPFileInstruction(new SimpleOperator(null), in, rl, ru, cl, cu, out, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else if ( parts[0].equalsIgnoreCase("leftIndex")) 
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
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		String opcode = getOpcode();
		long rl = ec.getScalarInput(rowLower.getName(), rowLower.getValueType(), rowLower.isLiteral()).getLongValue();
		long ru = ec.getScalarInput(rowUpper.getName(), rowUpper.getValueType(), rowUpper.isLiteral()).getLongValue();
		long cl = ec.getScalarInput(colLower.getName(), colLower.getValueType(), colLower.isLiteral()).getLongValue();
		long cu = ec.getScalarInput(colUpper.getName(), colUpper.getValueType(), colUpper.isLiteral()).getLongValue();
		MatrixObject mo = (MatrixObject) ec.getVariable(input1.getName());
		
		if( mo.isPartitioned() && opcode.equalsIgnoreCase("rangeReIndex") ) 
		{
			MatrixFormatMetaData meta = (MatrixFormatMetaData)mo.getMetaData();
			MatrixCharacteristics mc = meta.getMatrixCharacteristics();
			String pfname = mo.getPartitionFileName( new IndexRange(rl,ru,cl,cu), mc.getRowsPerBlock(), mc.getColsPerBlock());
			
			if( MapReduceTool.existsFileOnHDFS(pfname) )
			{
				MatrixObject out = (MatrixObject)ec.getVariable(output.getName());
				
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
					case COLUMN_WISE:
						mcNew = new MatrixCharacteristics( mc.getRows(), 1, mc.getRowsPerBlock(), mc.getColsPerBlock() );
						break;					
					default:
						throw new DMLRuntimeException("Unsupported partition format for CP_FILE rangeReIndex: "+ mo.getPartitionFormat());
				}
				
				MatrixFormatMetaData metaNew = new MatrixFormatMetaData(mcNew,meta.getOutputInfo(),meta.getInputInfo());
				mobj.setMetaData(metaNew);	 
				
				//put output object into symbol table
				ec.setVariable(output.getName(), mobj);
			}
			else
			{
				//will return an empty matrix partition 
				MatrixBlock resultBlock = mo.readMatrixPartition( new IndexRange(rl,ru,cl,cu) );
				ec.setMatrixOutput(output.getName(), resultBlock);
			}
		}
		else
		{
			throw new DMLRuntimeException("Invalid opcode or index predicate for MatrixIndexingCPFileInstruction: " + instString);	
		}
	}
}