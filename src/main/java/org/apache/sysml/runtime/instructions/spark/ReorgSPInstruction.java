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

package org.apache.sysml.runtime.instructions.spark;

import java.util.ArrayList;
import java.util.Iterator;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.functionobjects.DiagIndex;
import org.apache.sysml.runtime.functionobjects.RevIndex;
import org.apache.sysml.runtime.functionobjects.SortIndex;
import org.apache.sysml.runtime.functionobjects.SwapIndex;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.functions.FilterDiagBlocksFunction;
import org.apache.sysml.runtime.instructions.spark.functions.IsBlockInList;
import org.apache.sysml.runtime.instructions.spark.functions.IsBlockInRange;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.instructions.spark.utils.RDDSortUtils;
import org.apache.sysml.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysml.runtime.instructions.spark.functions.ReorgMapFunction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.IndexRange;
import org.apache.sysml.runtime.util.UtilFunctions;

public class ReorgSPInstruction extends UnarySPInstruction {
	// sort-specific attributes (to enable variable attributes)
	private CPOperand _col = null;
	private CPOperand _desc = null;
	private CPOperand _ixret = null;
	private boolean _bSortIndInMem = false;

	private ReorgSPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr) {
		super(SPType.Reorg, op, in, out, opcode, istr);
	}

	private ReorgSPInstruction(Operator op, CPOperand in, CPOperand col, CPOperand desc, CPOperand ixret, CPOperand out,
			String opcode, boolean bSortIndInMem, String istr) {
		this(op, in, out, opcode, istr);
		_col = col;
		_desc = desc;
		_ixret = ixret;
		_bSortIndInMem = bSortIndInMem;
	}

	public static ReorgSPInstruction parseInstruction ( String str ) {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = InstructionUtils.getOpCode(str);
		
		if ( opcode.equalsIgnoreCase("r'") ) {
			parseUnaryInstruction(str, in, out); //max 2 operands
			return new ReorgSPInstruction(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out, opcode, str);
		}
		else if ( opcode.equalsIgnoreCase("rev") ) {
			parseUnaryInstruction(str, in, out); //max 2 operands
			return new ReorgSPInstruction(new ReorgOperator(RevIndex.getRevIndexFnObject()), in, out, opcode, str);
		}
		else if ( opcode.equalsIgnoreCase("rdiag") ) {
			parseUnaryInstruction(str, in, out); //max 2 operands
			return new ReorgSPInstruction(new ReorgOperator(DiagIndex.getDiagIndexFnObject()), in, out, opcode, str);
		} 
		else if ( opcode.equalsIgnoreCase("rsort") ) {
			String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
			InstructionUtils.checkNumFields(parts, 5, 6);
			in.split(parts[1]);
			out.split(parts[5]);
			CPOperand col = new CPOperand(parts[2]);
			CPOperand desc = new CPOperand(parts[3]);
			CPOperand ixret = new CPOperand(parts[4]);
			boolean bSortIndInMem = false;
			
			if(parts.length > 5)
				bSortIndInMem = Boolean.parseBoolean(parts[6]);
			
			return new ReorgSPInstruction(new ReorgOperator(new SortIndex(1,false,false)),
				in, col, desc, ixret, out, opcode, bSortIndInMem, str);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ReorgInstruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		String opcode = getOpcode();

		//get input rdd handle
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		MatrixCharacteristics mcIn = sec.getMatrixCharacteristics(input1.getName());
		
		if( opcode.equalsIgnoreCase("r'") ) //TRANSPOSE
		{
			//execute transpose reorg operation
			out = in1.mapToPair(new ReorgMapFunction(opcode));
		}
		else if( opcode.equalsIgnoreCase("rev") ) //REVERSE
		{
			//execute reverse reorg operation
			out = in1.flatMapToPair(new RDDRevFunction(mcIn));
			if( mcIn.getRows() % mcIn.getRowsPerBlock() != 0 )
				out = RDDAggregateUtils.mergeByKey(out, false);
		}
		else if ( opcode.equalsIgnoreCase("rdiag") ) // DIAG
		{	
			if(mcIn.getCols() == 1) { // diagV2M
				out = in1.flatMapToPair(new RDDDiagV2MFunction(mcIn));
			}
			else { // diagM2V
				//execute diagM2V operation
				out = in1.filter(new FilterDiagBlocksFunction())
					     .mapToPair(new ReorgMapFunction(opcode));
			}
		}
		else if ( opcode.equalsIgnoreCase("rsort") ) //ORDER
		{
			// Sort by column 'col' in ascending/descending order and return either index/value
			
			//get parameters
			long[] cols = _col.getDataType().isMatrix() ? DataConverter.convertToLongVector(ec.getMatrixInput(_col.getName())) :
				new long[]{ec.getScalarInput(_col.getName(), _col.getValueType(), _col.isLiteral()).getLongValue()};
			boolean desc = ec.getScalarInput(_desc.getName(), _desc.getValueType(), _desc.isLiteral()).getBooleanValue();
			boolean ixret = ec.getScalarInput(_ixret.getName(), _ixret.getValueType(), _ixret.isLiteral()).getBooleanValue();
			boolean singleCol = (mcIn.getCols() == 1);
			out = in1;
			
			if( cols.length > mcIn.getColsPerBlock() ) 
				LOG.warn("Unsupported sort with number of order-by columns large than blocksize: "+cols.length);
			
			if( singleCol || cols.length==1 ) {
				// extract column (if necessary) and sort 
				if( !singleCol )
					out = out.filter(new IsBlockInRange(1, mcIn.getRows(), cols[0], cols[0], mcIn))
						.mapValues(new ExtractColumn((int)UtilFunctions.computeCellInBlock(cols[0], mcIn.getColsPerBlock())));
				
				//actual index/data sort operation
				if( ixret ) //sort indexes 
					out = RDDSortUtils.sortIndexesByVal(out, !desc, mcIn.getRows(), mcIn.getRowsPerBlock());
				else if( singleCol && !desc) //sort single-column matrix
					out = RDDSortUtils.sortByVal(out, mcIn.getRows(), mcIn.getRowsPerBlock());
				else if( !_bSortIndInMem ) //sort multi-column matrix w/ rewrite
					out = RDDSortUtils.sortDataByVal(out, in1, !desc, mcIn.getRows(), mcIn.getCols(), mcIn.getRowsPerBlock(), mcIn.getColsPerBlock());
				else //sort multi-column matrix
					out = RDDSortUtils.sortDataByValMemSort(out, in1, !desc, mcIn.getRows(), mcIn.getCols(), mcIn.getRowsPerBlock(), mcIn.getColsPerBlock(), sec, (ReorgOperator) _optr);
			}
			else { //general case: multi-column sort
				// extract columns (if necessary)
				if( cols.length < mcIn.getCols() )
					out = out.filter(new IsBlockInList(cols, mcIn))
						.mapToPair(new ExtractColumns(cols, mcIn));
				
				// append extracted columns (if necessary)
				if( mcIn.getCols() > mcIn.getColsPerBlock() )
					out = RDDAggregateUtils.mergeByKey(out);
				
				//actual index/data sort operation
				if( ixret ) //sort indexes 
					out = RDDSortUtils.sortIndexesByVals(out, !desc, mcIn.getRows(), (long)cols.length, mcIn.getRowsPerBlock());
				else if( cols.length==mcIn.getCols() && !desc) //sort single-column matrix
					out = RDDSortUtils.sortByVals(out, mcIn.getRows(), cols.length, mcIn.getRowsPerBlock());
				else //sort multi-column matrix
					out = RDDSortUtils.sortDataByVals(out, in1, !desc, mcIn.getRows(), mcIn.getCols(),
						cols.length, mcIn.getRowsPerBlock(), mcIn.getColsPerBlock());
			}
		}
		else {
			throw new DMLRuntimeException("Error: Incorrect opcode in ReorgSPInstruction:" + opcode);
		}
		
		//store output rdd handle
		if( opcode.equalsIgnoreCase("rsort") && _col.getDataType().isMatrix() )
			sec.releaseMatrixInput(_col.getName());
		updateReorgMatrixCharacteristics(sec);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
	}

	private void updateReorgMatrixCharacteristics(SparkExecutionContext sec) {
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		
		//infer initially unknown dimensions from inputs
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
		
		//infer initially unknown nnz from input
		if( !mcOut.nnzKnown() && mc1.nnzKnown() ){
			boolean sortIx = getOpcode().equalsIgnoreCase("rsort") && sec.getScalarInput(_ixret.getName(), _ixret.getValueType(), _ixret.isLiteral()).getBooleanValue();			
			if( sortIx )
				mcOut.setNonZeros(mc1.getRows());
			else //default (r', rdiag, rsort data)
				mcOut.setNonZeros(mc1.getNonZeros());
		}
	}

	private static class RDDDiagV2MFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 31065772250744103L;
		
		private ReorgOperator _reorgOp = null;
		private MatrixCharacteristics _mcIn = null;
		
		public RDDDiagV2MFunction(MatrixCharacteristics mcIn) {
			_reorgOp = new ReorgOperator(DiagIndex.getDiagIndexFnObject());
			_mcIn = mcIn;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) {
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> ret = new ArrayList<>();
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();
			
			//compute output indexes and reorg data
			long rix = ixIn.getRowIndex();
			MatrixIndexes ixOut = new MatrixIndexes(rix, rix);
			MatrixBlock blkOut = (MatrixBlock) blkIn.reorgOperations(_reorgOp, new MatrixBlock(), -1, -1, -1);
			ret.add(new Tuple2<>(ixOut,blkOut));
			
			// insert newly created empty blocks for entire row
			int numBlocks = (int) Math.ceil((double)_mcIn.getRows()/_mcIn.getRowsPerBlock());
			for(int i = 1; i <= numBlocks; i++) {
				if(i != ixOut.getColumnIndex()) {
					int lrlen = UtilFunctions.computeBlockSize(_mcIn.getRows(), rix, _mcIn.getRowsPerBlock());
					int lclen = UtilFunctions.computeBlockSize(_mcIn.getRows(), i, _mcIn.getRowsPerBlock());
					MatrixBlock emptyBlk = new MatrixBlock(lrlen, lclen, true);
					ret.add(new Tuple2<>(new MatrixIndexes(rix, i), emptyBlk));
				}
			}
			
			return ret.iterator();
		}
	}

	private static class RDDRevFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 1183373828539843938L;
		
		private MatrixCharacteristics _mcIn = null;
		
		public RDDRevFunction(MatrixCharacteristics mcIn) {
			_mcIn = mcIn;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) {
			//construct input
			IndexedMatrixValue in = SparkUtils.toIndexedMatrixBlock(arg0);
			
			//execute reverse operation
			ArrayList<IndexedMatrixValue> out = new ArrayList<>();
			LibMatrixReorg.rev(in, _mcIn.getRows(), _mcIn.getRowsPerBlock(), out);
			
			//construct output
			return SparkUtils.fromIndexedMatrixBlock(out).iterator();
		}
	}

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
			return arg0.slice(0, arg0.getNumRows()-1, _col, _col, new MatrixBlock());
		}
	}
	
	private static class ExtractColumns implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>,MatrixIndexes,MatrixBlock>
	{
		private static final long serialVersionUID = 2902729186431711506L;
		
		private final long[] _cols;
		private final int _brlen, _bclen;
		
		public ExtractColumns(long[] cols, MatrixCharacteristics mc) {
			_cols = cols;
			_brlen = mc.getRowsPerBlock();
			_bclen = mc.getColsPerBlock();
		}
		
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) {
			MatrixIndexes ix = arg0._1();
			MatrixBlock in = arg0._2();
			MatrixBlock out = new MatrixBlock(in.getNumRows(), _cols.length, true);
			for(int i=0; i<_cols.length; i++)
				if( UtilFunctions.isInBlockRange(ix, _brlen, _bclen, new IndexRange(1, Long.MAX_VALUE, _cols[i], _cols[i])) ) {
					int index = UtilFunctions.computeCellInBlock(_cols[i], _bclen);
					out.leftIndexingOperations(in.slice(0, in.getNumRows()-1, index, index, new MatrixBlock()),
						0, in.getNumRows()-1, i, i, out, UpdateType.INPLACE);
				}
			return new Tuple2<>(new MatrixIndexes(ix.getRowIndex(), 1), out);
		}
	}
}

