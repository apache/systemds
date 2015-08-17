/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import java.util.ArrayList;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.DiagIndex;
import com.ibm.bi.dml.runtime.functionobjects.SortIndex;
import com.ibm.bi.dml.runtime.functionobjects.SwapIndex;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.functions.FilterDiagBlocksFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.IsBlockInRange;
import com.ibm.bi.dml.runtime.instructions.spark.utils.RDDSortUtils;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ReorgMapFunction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class ReorgSPInstruction extends UnarySPInstruction
{
	
	//sort-specific attributes (to enable variable attributes)
 	private CPOperand _col = null;
 	private CPOperand _desc = null;
 	private CPOperand _ixret = null;
	 	
	public ReorgSPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr){
		super(op, in, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.Reorg;
	}
	
	public ReorgSPInstruction(Operator op, CPOperand in, CPOperand col, CPOperand desc, CPOperand ixret, CPOperand out, String opcode, String istr){
		this(op, in, out, opcode, istr);
		_col = col;
		_desc = desc;
		_ixret = ixret;
		_sptype = SPINSTRUCTION_TYPE.Reorg;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = InstructionUtils.getOpCode(str);
		
		if ( opcode.equalsIgnoreCase("r'") ) {
			parseUnaryInstruction(str, in, out); //max 2 operands
			return new ReorgSPInstruction(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out, opcode, str);
		} 
		else if ( opcode.equalsIgnoreCase("rdiag") ) {
			parseUnaryInstruction(str, in, out); //max 2 operands
			return new ReorgSPInstruction(new ReorgOperator(DiagIndex.getDiagIndexFnObject()), in, out, opcode, str);
		} 
		else if ( opcode.equalsIgnoreCase("rsort") ) {
			InstructionUtils.checkNumFields(str, 5);
			String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
			in.split(parts[1]);
			out.split(parts[5]);
			CPOperand col = new CPOperand(parts[2]);
			CPOperand desc = new CPOperand(parts[3]);
			CPOperand ixret = new CPOperand(parts[4]);
			
			return new ReorgSPInstruction(new ReorgOperator(SortIndex.getSortIndexFnObject(1,false,false)), 
					                      in, col, desc, ixret, out, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ReorgInstruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		String opcode = getOpcode();
		
		if( opcode.equalsIgnoreCase("r'") ) //TRANSPOSE
		{
			//get input rdd handle
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );

			//execute transpose reorg operation
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1.mapToPair(new ReorgMapFunction(opcode));
			
			//store output rdd handle
			updateReorgMatrixCharacteristics(sec);
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else if ( opcode.equalsIgnoreCase("rdiag") ) // DIAG
		{			
			//update and get matrix characteristics
			MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			updateReorgMatrixCharacteristics(sec); //update mcOut
			
			// get input rdd handle
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
			if(mc1.getCols() == 1) { // diagV2M
				out = in1.flatMapToPair(new RDDDiagV2MFunction(mcOut));
			}
			else { // diagM2V
				//execute diagM2V operation
				out = in1.filter(new FilterDiagBlocksFunction())
					     .mapToPair(new ReorgMapFunction(opcode));
			}
			
			//store output rdd handle
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else if ( opcode.equalsIgnoreCase("rsort") ) //ORDER
		{
			// Sort by column 'col' in ascending/descending order and return either index/value
			
			//get parameters
			long col = ec.getScalarInput(_col.getName(), _col.getValueType(), _col.isLiteral()).getLongValue();
			boolean desc = ec.getScalarInput(_desc.getName(), _desc.getValueType(), _desc.isLiteral()).getBooleanValue();
			boolean ixret = ec.getScalarInput(_ixret.getName(), _ixret.getValueType(), _ixret.isLiteral()).getBooleanValue();
			
			// get input rdd handle
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
			MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
			boolean singleCol = (mc1.getCols() == 1);
			
			// extract column (if necessary) and sort 
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1;
			if( !singleCol ){
				out = out.filter(new IsBlockInRange(1, mc1.getRows(), col, col, mc1.getRowsPerBlock(), mc1.getColsPerBlock()))
						 .mapValues(new ExtractColumn((int)UtilFunctions.cellInBlockCalculation(col, mc1.getColsPerBlock())));
			}
			
			//actual index/data sort operation
			if( ixret ) { //sort indexes 
				out = RDDSortUtils.sortIndexesByVal(out, !desc, mc1.getRows(), mc1.getRowsPerBlock());
			}	
			else if( singleCol && !desc) { //sort single-column matrix
				out = RDDSortUtils.sortByVal(out, mc1.getRows(), mc1.getRowsPerBlock());
			}
			else { //sort multi-column matrix
				out = RDDSortUtils.sortDataByVal(out, in1, !desc, mc1.getRows(), mc1.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock());
			}
			
			//store output rdd handle
			updateReorgMatrixCharacteristics(sec);
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else {
			throw new DMLRuntimeException("Error: Incorrect opcode in ReorgSPInstruction:" + opcode);
		}
	}
	
	/**
	 * 
	 * @param sec
	 * @throws DMLRuntimeException
	 */
	private void updateReorgMatrixCharacteristics(SparkExecutionContext sec) 
		throws DMLRuntimeException
	{
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		
		if( !mcOut.dimsKnown() ) 
		{
			if( !mc1.dimsKnown() )
				throw new DMLRuntimeException("Unable to compute output matrix characteristics from input.");
			
			if ( getOpcode().equalsIgnoreCase("r'") ) 
				mcOut.set(mc1.getCols(), mc1.getRows(), mc1.getColsPerBlock(), mc1.getRowsPerBlock());
			else if ( getOpcode().equalsIgnoreCase("rdiag") )
				mcOut.set(mc1.getRows(), (mc1.getCols()>1)?1:mc1.getRows(), mc1.getRowsPerBlock(), mc1.getColsPerBlock());
			else if ( getOpcode().equalsIgnoreCase("rsort") ) {
				boolean ixret = sec.getScalarInput(_ixret.getName(), _ixret.getValueType(), _ixret.isLiteral()).getBooleanValue();
				mcOut.set(mc1.getRows(), ixret?1:mc1.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock());
			}
		}
	}
	
	/**
	 * 
	 */
	private static class RDDDiagV2MFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 31065772250744103L;
		
		private ReorgOperator _reorgOp = null;
		private MatrixCharacteristics _mcOut = null;
		
		public RDDDiagV2MFunction(MatrixCharacteristics mcOut) 
			throws DMLRuntimeException 
		{
			_reorgOp = new ReorgOperator(DiagIndex.getDiagIndexFnObject());
			_mcOut = mcOut;
		}
		
		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
			
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();
			
			//compute output indexes and reorg data
			long rix = ixIn.getRowIndex();
			MatrixIndexes ixOut = new MatrixIndexes(rix, rix);
			MatrixBlock blkOut = (MatrixBlock) blkIn.reorgOperations(_reorgOp, new MatrixBlock(), -1, -1, -1);
			ret.add(new Tuple2<MatrixIndexes, MatrixBlock>(ixOut,blkOut));
			
			// insert newly created empty blocks for entire row
			int numBlocks = (int) Math.ceil((double)_mcOut.getCols()/_mcOut.getColsPerBlock());
			for(int i = 1; i <= numBlocks; i++) {
				if(i != ixOut.getColumnIndex()) {
					int lrlen = UtilFunctions.computeBlockSize(_mcOut.getRows(), rix, _mcOut.getRowsPerBlock());
		    		int lclen = UtilFunctions.computeBlockSize(_mcOut.getCols(), i, _mcOut.getColsPerBlock());
		    		MatrixBlock emptyBlk = new MatrixBlock(lrlen, lclen, true);
					ret.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(rix, i), emptyBlk));
				}
			}
			
			return ret;
		}
	}

	/**
	 *
	 */
	private static class ExtractColumn implements Function<MatrixBlock, MatrixBlock>  
	{
		private static final long serialVersionUID = -1472164797288449559L;
		
		private int _col;
		
		public ExtractColumn(int col) {
			_col = col;
		}
		
		@Override
		public MatrixBlock call(MatrixBlock arg0) 
			throws Exception 
		{
			return arg0.sliceOperations(1, arg0.getNumRows(), _col+1, _col+1, new MatrixBlock());
		}
	}
}

