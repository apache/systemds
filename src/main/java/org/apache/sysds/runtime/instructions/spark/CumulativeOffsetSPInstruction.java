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

package org.apache.sysds.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysds.runtime.matrix.data.LibMatrixAgg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;

public class CumulativeOffsetSPInstruction extends BinarySPInstruction {
	private UnaryOperator _uop = null;
	private boolean _cumsumprod = false;
	private final double _initValue ;
	private final boolean _broadcast;

	private CumulativeOffsetSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, double init, boolean broadcast, String opcode, String istr) {
		super(SPType.CumsumOffset, op, in1, in2, out, opcode, istr);

		if (Opcodes.BCUMOFFKP.toString().equals(opcode))
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucumk+"));
		else if (Opcodes.BROWCUMOFFKP.toString().equals(opcode))
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("urowcumk+"));
		else if (Opcodes.BCUMOFFM.toString().equals(opcode))
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucum*"));
		else if (Opcodes.BCUMOFFPM.toString().equals(opcode)) {
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucumk+*"));
			_cumsumprod = true;
		}
		else if (Opcodes.BCUMOFFMIN.toString().equals(opcode))
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucummin"));
		else if (Opcodes.BCUMOFFMAX.toString().equals(opcode))
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucummax"));

		_initValue = init;
		_broadcast = broadcast;
	}

	public static CumulativeOffsetSPInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType( str );
		InstructionUtils.checkNumFields(parts, 5);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[3]);
		double init = Double.parseDouble(parts[4]);
		boolean broadcast = Boolean.parseBoolean(parts[5]);
		return new CumulativeOffsetSPInstruction(null, in1, in2, out, init, broadcast, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		DataCharacteristics mc1 = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mc2 = sec.getDataCharacteristics(input2.getName());
		long rlen = mc2.getRows();
		int blen = mc2.getBlocksize();
		
		//get and join inputs
		JavaPairRDD<MatrixIndexes,MatrixBlock> inData = sec.getBinaryMatrixBlockRDDHandleForVariable(input1.getName());
		JavaPairRDD<MatrixIndexes,Tuple2<MatrixBlock,MatrixBlock>> joined = null;
		boolean broadcast = _broadcast && !SparkUtils.isHashPartitioned(inData);
		
		if( broadcast ) {
			//broadcast offsets and broadcast join with data
			PartitionedBroadcast<MatrixBlock> inAgg = sec.getBroadcastForVariable(input2.getName());
			joined = inData.mapToPair(new RDDCumSplitLookupFunction(inAgg,_initValue, rlen, blen));
		}
		else {
			//prepare aggregates (cumsplit of offsets) and repartition join with data
			joined = inData.join(sec
				.getBinaryMatrixBlockRDDHandleForVariable(input2.getName())
				.flatMapToPair(new RDDCumSplitFunction(_initValue, rlen, blen)));
		}
		
		//execute cumulative offset (apply cumulative op w/ offsets)
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = joined
			.mapValues(new RDDCumOffsetFunction(_uop, _cumsumprod));
		
		//put output handle in symbol table
		if( _cumsumprod )
			sec.getDataCharacteristics(output.getName())
				.set(mc1.getRows(), 1, mc1.getBlocksize(), mc1.getBlocksize());
		else //general case
			updateUnaryOutputDataCharacteristics(sec);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		sec.addLineage(output.getName(), input2.getName(), broadcast);
	}

	public double getInitValue() {
		return _initValue;
	}

	public boolean getBroadcast() {
		return _broadcast;
	}

	private static class RDDCumSplitFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -8407407527406576965L;
		
		private double _initValue = 0;
		private int _blen = -1;
		private long _lastRowBlockIndex;
		
		public RDDCumSplitFunction( double initValue, long rlen, int blen )
		{
			_initValue = initValue;
			_blen = blen;
			_lastRowBlockIndex = (long)Math.ceil((double)rlen/blen);
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> ret = new ArrayList<>();
			
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();
			
			long rixOffset = (ixIn.getRowIndex()-1)*_blen;
			boolean firstBlk = (ixIn.getRowIndex() == 1);
			boolean lastBlk = (ixIn.getRowIndex() == _lastRowBlockIndex );
			
			//introduce offsets w/ init value for first row 
			if( firstBlk ) { 
				MatrixIndexes tmpix = new MatrixIndexes(1, ixIn.getColumnIndex());
				MatrixBlock tmpblk = new MatrixBlock(1, blkIn.getNumColumns(), blkIn.isInSparseFormat());
				if( _initValue != 0 ){
					for( int j=0; j<blkIn.getNumColumns(); j++ )
						tmpblk.appendValue(0, j, _initValue);
				}
				ret.add(new Tuple2<>(tmpix, tmpblk));
			}
			
			//output splitting (shift by one), preaggregated offset used by subsequent block
			for( int i=0; i<blkIn.getNumRows(); i++ )
				if( !(lastBlk && i==(blkIn.getNumRows()-1)) ) //ignore last row
				{
					MatrixIndexes tmpix = new MatrixIndexes(rixOffset+i+2, ixIn.getColumnIndex());
					MatrixBlock tmpblk = new MatrixBlock(1, blkIn.getNumColumns(), blkIn.isInSparseFormat());
					blkIn.slice(i, i, 0, blkIn.getNumColumns()-1, tmpblk);
					ret.add(new Tuple2<>(tmpix, tmpblk));
				}
			
			return ret.iterator();
		}
	}
	
	private static class RDDCumSplitLookupFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, Tuple2<MatrixBlock,MatrixBlock>> 
	{
		private static final long serialVersionUID = -2785629043886477479L;
		
		private final PartitionedBroadcast<MatrixBlock> _pbc;
		private final double _initValue;
		private final int _blen;
		
		public RDDCumSplitLookupFunction(PartitionedBroadcast<MatrixBlock> pbc, double initValue, long rlen, int blen) {
			_pbc = pbc;
			_initValue = initValue;
			_blen = blen;
		}
		
		@Override
		public Tuple2<MatrixIndexes, Tuple2<MatrixBlock,MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) throws Exception {
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();
			
			//compute block and row indexes
			long brix = UtilFunctions.computeBlockIndex(ixIn.getRowIndex()-1, _blen);
			int rix = UtilFunctions.computeCellInBlock(ixIn.getRowIndex()-1, _blen);
			
			//lookup offset row and return joined output
			MatrixBlock off = (ixIn.getRowIndex() == 1) ? new MatrixBlock(1, blkIn.getNumColumns(), _initValue) :
				_pbc.getBlock((int)brix, (int)ixIn.getColumnIndex()).slice(rix, rix);
			return new Tuple2<>(ixIn, new Tuple2<>(blkIn,off));
		}
	}

	private static class RDDCumOffsetFunction implements Function<Tuple2<MatrixBlock, MatrixBlock>, MatrixBlock> 
	{
		private static final long serialVersionUID = -5804080263258064743L;

		private final UnaryOperator _uop;
		private final boolean _cumsumprod;
		
		public RDDCumOffsetFunction(UnaryOperator uop, boolean cumsumprod) {
			_uop = uop;
			_cumsumprod = cumsumprod;
		}

		@Override
		public MatrixBlock call(Tuple2<MatrixBlock, MatrixBlock> arg0) throws Exception  {
			//prepare inputs and outputs
			MatrixBlock dblkIn = arg0._1(); //original data 
			MatrixBlock oblkIn = arg0._2(); //offset row vector
			
			//allocate output block
			MatrixBlock blkOut = new MatrixBlock(dblkIn.getNumRows(),
				_cumsumprod ? 1 : dblkIn.getNumColumns(), false);
			
			//blockwise cumagg computation, incl offset aggregation
			return LibMatrixAgg.cumaggregateUnaryMatrix(dblkIn, blkOut, _uop,
				DataConverter.convertToDoubleVector(oblkIn, false,
				((Builtin)_uop.fn).bFunc == BuiltinCode.CUMSUM));
		}
	}
}
