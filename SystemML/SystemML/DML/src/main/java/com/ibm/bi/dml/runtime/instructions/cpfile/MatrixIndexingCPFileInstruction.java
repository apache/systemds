/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cpfile;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.cp.MatrixIndexingCPInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.RangeBasedReIndexInstruction.IndexRange;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public MatrixIndexingCPFileInstruction(Operator op, CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String istr)
	{
		super( op, in, rl, ru, cl, cu, out, istr );
	}
	
	public MatrixIndexingCPFileInstruction(Operator op, CPOperand lhsInput, CPOperand rhsInput, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String istr)
	{
		super( op, lhsInput, rhsInput, rl, ru, cl, cu, out, istr);
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		
		if ( parts[0].equalsIgnoreCase("rangeReIndex") ) {
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
				return new MatrixIndexingCPFileInstruction(new SimpleOperator(null), in, rl, ru, cl, cu, out, str);
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
		String opcode = InstructionUtils.getOpCode( instString );
		long rl = ec.getScalarInput(rowLower.get_name(), rowLower.get_valueType(), rowLower.isLiteral()).getLongValue();
		long ru = ec.getScalarInput(rowUpper.get_name(), rowUpper.get_valueType(), rowUpper.isLiteral()).getLongValue();
		long cl = ec.getScalarInput(colLower.get_name(), colLower.get_valueType(), colLower.isLiteral()).getLongValue();
		long cu = ec.getScalarInput(colUpper.get_name(), colUpper.get_valueType(), colUpper.isLiteral()).getLongValue();
		MatrixObject mo = (MatrixObject) ec.getVariable(input1.get_name());
		
		if( mo.isPartitioned() && opcode.equalsIgnoreCase("rangeReIndex") ) 
		{
			MatrixFormatMetaData meta = (MatrixFormatMetaData)mo.getMetaData();
			MatrixCharacteristics mc = meta.getMatrixCharacteristics();
			String pfname = mo.getPartitionFileName( new IndexRange(rl,ru,cl,cu), mc.get_rows_per_block(), mc.get_cols_per_block());
			
			if( MapReduceTool.existsFileOnHDFS(pfname) )
			{
				MatrixObject out = (MatrixObject)ec.getVariable(output.get_name());
				
				//create output matrix object				
				MatrixObject mobj = new MatrixObject(mo.getValueType(), pfname );
				mobj.setDataType( DataType.MATRIX );
				mobj.setVarName( out.getVarName() );
				MatrixCharacteristics mcNew = null;
				switch( mo.getPartitionFormat() )
				{
					case ROW_WISE:
						mcNew = new MatrixCharacteristics( 1, mc.get_cols(), mc.get_rows_per_block(), mc.get_cols_per_block() );
						break;
					case COLUMN_WISE:
						mcNew = new MatrixCharacteristics( mc.get_rows(), 1, mc.get_rows_per_block(), mc.get_cols_per_block() );
						break;					
					default:
						throw new DMLRuntimeException("Unsupported partition format for CP_FILE rangeReIndex: "+ mo.getPartitionFormat());
				}
				
				MatrixFormatMetaData metaNew = new MatrixFormatMetaData(mcNew,meta.getOutputInfo(),meta.getInputInfo());
				mobj.setMetaData(metaNew);	 
				
				//put output object into symbol table
				ec.setVariable(output.get_name(), mobj);
			}
			else
			{
				//will return an empty matrix partition 
				MatrixBlock resultBlock = mo.readMatrixPartition( new IndexRange(rl,ru,cl,cu) );
				ec.setMatrixOutput(output.get_name(), resultBlock);
			}
		}
		else
		{
			throw new DMLRuntimeException("Invalid opcode or index predicate for MatrixIndexingCPFileInstruction: " + instString);	
		}
	}
}