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

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.functionobjects.DiagIndex;
import org.tugraz.sysds.runtime.functionobjects.RevIndex;
import org.tugraz.sysds.runtime.functionobjects.SortIndex;
import org.tugraz.sysds.runtime.functionobjects.SwapIndex;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.instructions.spark.functions.FilterDiagMatrixBlocksFunction;
import org.tugraz.sysds.runtime.instructions.spark.functions.IsBlockInList;
import org.tugraz.sysds.runtime.instructions.spark.functions.IsBlockInRange;
import org.tugraz.sysds.runtime.instructions.spark.functions.ReorgMapFunction;
import org.tugraz.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.tugraz.sysds.runtime.instructions.spark.utils.RDDSortUtils;
import org.tugraz.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.tugraz.sysds.runtime.matrix.data.LibMatrixReorg;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.mapred.IndexedMatrixValue;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.matrix.operators.ReorgOperator;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.runtime.util.IndexRange;
import org.tugraz.sysds.runtime.util.UtilFunctions;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;

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
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		DataCharacteristics mcIn = sec.getDataCharacteristics(input1.getName());
		
		if( opcode.equalsIgnoreCase("r'") ) //TRANSPOSE
		{
			//execute transpose reorg operation
			out = in1.mapToPair(new ReorgMapFunction(opcode));
		}
		else if( opcode.equalsIgnoreCase("rev") ) //REVERSE
		{
			//execute reverse reorg operation
			out = in1.flatMapToPair(new RDDRevFunction(mcIn));
			if( mcIn.getRows() % mcIn.getBlocksize() != 0 )
				out = RDDAggregateUtils.mergeByKey(out, false);
		}
		else if ( opcode.equalsIgnoreCase("rdiag") ) // DIAG
		{	
			if(mcIn.getCols() == 1) { // diagV2M
				out = in1.flatMapToPair(new RDDDiagV2MFunction(mcIn));
			}
			else { // diagM2V
				//execute diagM2V operation
				out = in1.filter(new FilterDiagMatrixBlocksFunction())
					     .mapToPair(new ReorgMapFunction(opcode));
			}
		}
		else if ( opcode.equalsIgnoreCase("rsort") ) //ORDER
		{
			// Sort by column 'col' in ascending/descending order and return either index/value
			
			//get parameters
			long[] cols = _col.getDataType().isMatrix() ? DataConverter.convertToLongVector(ec.getMatrixInput(_col.getName())) :
				new long[]{ec.getScalarInput(_col).getLongValue()};
			boolean desc = ec.getScalarInput(_desc).getBooleanValue();
			boolean ixret = ec.getScalarInput(_ixret).getBooleanValue();
			boolean singleCol = (mcIn.getCols() == 1);
			out = in1;
			
			if( cols.length > mcIn.getBlocksize() ) 
				LOG.warn("Unsupported sort with number of order-by columns large than blocksize: "+cols.length);
			
			if( singleCol || cols.length==1 ) {
				// extract column (if necessary) and sort 
				if( !singleCol )
					out = out.filter(new IsBlockInRange(1, mcIn.getRows(), cols[0], cols[0], mcIn))
						.mapValues(new ExtractColumn((int)UtilFunctions.computeCellInBlock(cols[0], mcIn.getBlocksize())));
				
				//actual index/data sort operation
				if( ixret ) //sort indexes 
					out = RDDSortUtils.sortIndexesByVal(out, !desc, mcIn.getRows(), mcIn.getBlocksize());
				else if( singleCol && !desc) //sort single-column matrix
					out = RDDSortUtils.sortByVal(out, mcIn.getRows(), mcIn.getBlocksize());
				else if( !_bSortIndInMem ) //sort multi-column matrix w/ rewrite
					out = RDDSortUtils.sortDataByVal(out, in1, !desc, mcIn.getRows(), mcIn.getCols(), mcIn.getBlocksize());
				else //sort multi-column matrix
					out = RDDSortUtils.sortDataByValMemSort(out, in1, !desc, mcIn.getRows(), mcIn.getCols(), mcIn.getBlocksize(), sec, (ReorgOperator) _optr);
			}
			else { //general case: multi-column sort
				// extract columns (if necessary)
				if( cols.length < mcIn.getCols() )
					out = out.filter(new IsBlockInList(cols, mcIn))
						.mapToPair(new ExtractColumns(cols, mcIn));
				
				// append extracted columns (if necessary)
				if( mcIn.getCols() > mcIn.getBlocksize() )
					out = RDDAggregateUtils.mergeByKey(out);
				
				//actual index/data sort operation
				if( ixret ) //sort indexes 
					out = RDDSortUtils.sortIndexesByVals(out, !desc, mcIn.getRows(), (long)cols.length, mcIn.getBlocksize());
				else if( cols.length==mcIn.getCols() && !desc) //sort single-column matrix
					out = RDDSortUtils.sortByVals(out, mcIn.getRows(), cols.length, mcIn.getBlocksize());
				else //sort multi-column matrix
					out = RDDSortUtils.sortDataByVals(out, in1, !desc, mcIn.getRows(),
						mcIn.getCols(), cols.length, mcIn.getBlocksize());
			}
		}
		else {
			throw new DMLRuntimeException("Error: Incorrect opcode in ReorgSPInstruction:" + opcode);
		}
		
		//store output rdd handle
		if( opcode.equalsIgnoreCase("rsort") && _col.getDataType().isMatrix() )
			sec.releaseMatrixInput(_col.getName());
		updateReorgDataCharacteristics(sec);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
	}

	private void updateReorgDataCharacteristics(SparkExecutionContext sec) {
		DataCharacteristics mc1 = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
		
		//infer initially unknown dimensions from inputs
		if( !mcOut.dimsKnown() ) 
		{
			if( !mc1.dimsKnown() )
				throw new DMLRuntimeException("Unable to compute output matrix characteristics from input.");
			
			if ( getOpcode().equalsIgnoreCase("r'") ) 
				mcOut.set(mc1.getCols(), mc1.getRows(), mc1.getBlocksize(), mc1.getBlocksize());
			else if ( getOpcode().equalsIgnoreCase("rdiag") )
				mcOut.set(mc1.getRows(), (mc1.getCols()>1)?1:mc1.getRows(), mc1.getBlocksize(), mc1.getBlocksize());
			else if ( getOpcode().equalsIgnoreCase("rsort") ) {
				boolean ixret = sec.getScalarInput(_ixret).getBooleanValue();
				mcOut.set(mc1.getRows(), ixret?1:mc1.getCols(), mc1.getBlocksize(), mc1.getBlocksize());
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
		private DataCharacteristics _mcIn = null;
		
		public RDDDiagV2MFunction(DataCharacteristics mcIn) {
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
			int numBlocks = (int) Math.ceil((double)_mcIn.getRows()/_mcIn.getBlocksize());
			for(int i = 1; i <= numBlocks; i++) {
				if(i != ixOut.getColumnIndex()) {
					int lrlen = UtilFunctions.computeBlockSize(_mcIn.getRows(), rix, _mcIn.getBlocksize());
					int lclen = UtilFunctions.computeBlockSize(_mcIn.getRows(), i, _mcIn.getBlocksize());
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
		
		private DataCharacteristics _mcIn = null;
		
		public RDDRevFunction(DataCharacteristics mcIn) {
			_mcIn = mcIn;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) {
			//construct input
			IndexedMatrixValue in = SparkUtils.toIndexedMatrixBlock(arg0);
			
			//execute reverse operation
			ArrayList<IndexedMatrixValue> out = new ArrayList<>();
			LibMatrixReorg.rev(in, _mcIn.getRows(), _mcIn.getBlocksize(), out);
			
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
		private final int _blen;
		
		public ExtractColumns(long[] cols, DataCharacteristics mc) {
			_cols = cols;
			_blen = mc.getBlocksize();
		}
		
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) {
			MatrixIndexes ix = arg0._1();
			MatrixBlock in = arg0._2();
			MatrixBlock out = new MatrixBlock(in.getNumRows(), _cols.length, true);
			for(int i=0; i<_cols.length; i++)
				if( UtilFunctions.isInBlockRange(ix, _blen, new IndexRange(1, Long.MAX_VALUE, _cols[i], _cols[i])) ) {
					int index = UtilFunctions.computeCellInBlock(_cols[i], _blen);
					out.leftIndexingOperations(in.slice(0, in.getNumRows()-1, index, index, new MatrixBlock()),
						0, in.getNumRows()-1, i, i, out, UpdateType.INPLACE);
				}
			return new Tuple2<>(new MatrixIndexes(ix.getRowIndex(), 1), out);
		}
	}
}

