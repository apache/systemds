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

import scala.Tuple2;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.UnaryOperator;


public class CumulativeOffsetSPInstruction extends BinarySPInstruction 
{

	private BinaryOperator _bop = null;
	private UnaryOperator _uop = null;
	private double _initValue = 0;
	
	public CumulativeOffsetSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, double init, String opcode, String istr)
	{
		super(op, in1, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.CumsumOffset;
		
		if( "bcumoffk+".equals(opcode) ) {
			_bop = new BinaryOperator(Plus.getPlusFnObject());
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucumk+"));
		}
		else if( "bcumoff*".equals(opcode) ){
			_bop = new BinaryOperator(Multiply.getMultiplyFnObject());
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucum*"));	
		}
		else if( "bcumoffmin".equals(opcode) ){
			_bop = new BinaryOperator(Builtin.getBuiltinFnObject("min"));
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucummin"));	
		}
		else if( "bcumoffmax".equals(opcode) ){
			_bop = new BinaryOperator(Builtin.getBuiltinFnObject("max"));
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucummax"));	
		}
		
		_initValue = init;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static CumulativeOffsetSPInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType( str );
		InstructionUtils.checkNumFields ( parts, 4 );
		
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[3]);
		double init = Double.parseDouble(parts[4]);
		
		return new CumulativeOffsetSPInstruction(null, in1, in2, out, init, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		MatrixCharacteristics mc = sec.getMatrixCharacteristics(input2.getName());
		long rlen = mc.getRows();
		int brlen = mc.getRowsPerBlock();
		
		//get inputs
		JavaPairRDD<MatrixIndexes,MatrixBlock> inData = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> inAgg = sec.getBinaryBlockRDDHandleForVariable( input2.getName() );
		
		//prepare aggregates (cumsplit of offsets)
		inAgg = inAgg.flatMapToPair(new RDDCumSplitFunction(_initValue, rlen, brlen));
		
		//execute cumulative offset (apply cumulative op w/ offsets)
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = 
				inData.join( inAgg )
				      .mapValues(new RDDCumOffsetFunction(_uop, _bop));
		
		updateUnaryOutputMatrixCharacteristics(sec);
		//put output handle in symbol table
		sec.setRDDHandleForVariable(output.getName(), out);	
		sec.addLineageRDD(output.getName(), input1.getName());
		sec.addLineageRDD(output.getName(), input2.getName());
	}
	
	
	/**
	 * 
	 * 
	 */
	private static class RDDCumSplitFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -8407407527406576965L;
		
		private double _initValue = 0;
		private int _brlen = -1;
		private long _lastRowBlockIndex;
		
		public RDDCumSplitFunction( double initValue, long rlen, int brlen )
		{
			_initValue = initValue;
			_brlen = brlen;
			_lastRowBlockIndex = (long)Math.ceil((double)rlen/brlen);
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes, MatrixBlock>>();
			
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();
			
			long rixOffset = (ixIn.getRowIndex()-1)*_brlen;
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
				ret.add(new Tuple2<MatrixIndexes,MatrixBlock>(tmpix, tmpblk));
			}	
			
			//output splitting (shift by one), preaggregated offset used by subsequent block
			for( int i=0; i<blkIn.getNumRows(); i++ )
				if( !(lastBlk && i==(blkIn.getNumRows()-1)) ) //ignore last row
				{
					MatrixIndexes tmpix = new MatrixIndexes(rixOffset+i+2, ixIn.getColumnIndex());
					MatrixBlock tmpblk = new MatrixBlock(1, blkIn.getNumColumns(), blkIn.isInSparseFormat());
					blkIn.sliceOperations(i, i, 0, blkIn.getNumColumns()-1, tmpblk);	
					ret.add(new Tuple2<MatrixIndexes,MatrixBlock>(tmpix, tmpblk));
				}
			
			return ret.iterator();
		}
	}
	
	/**
	 * 
	 */
	private static class RDDCumOffsetFunction implements Function<Tuple2<MatrixBlock, MatrixBlock>, MatrixBlock> 
	{
		private static final long serialVersionUID = -5804080263258064743L;

		private UnaryOperator _uop = null;
		private BinaryOperator _bop = null;
		
		public RDDCumOffsetFunction(UnaryOperator uop, BinaryOperator bop)
		{
			_uop = uop;
			_bop = bop;
		}

		@Override
		public MatrixBlock call(Tuple2<MatrixBlock, MatrixBlock> arg0)
			throws Exception 
		{
			//prepare inputs and outputs
			MatrixBlock dblkIn = arg0._1(); //original data 
			MatrixBlock oblkIn = arg0._2(); //offset row vector
			MatrixBlock blkOut = new MatrixBlock(dblkIn.getNumRows(), dblkIn.getNumColumns(), dblkIn.isInSparseFormat());
			
			//blockwise offset aggregation and prefix sum computation
			MatrixBlock data2 = new MatrixBlock(dblkIn); //cp data
			MatrixBlock fdata2 = data2.sliceOperations(0, 0, 0, data2.getNumColumns()-1, new MatrixBlock()); //1-based
			fdata2.binaryOperationsInPlace(_bop, oblkIn); //sum offset to first row
			data2.copy(0, 0, 0, data2.getNumColumns()-1, fdata2, true); //0-based
			data2.unaryOperations(_uop, blkOut); //compute columnwise prefix sums/prod/min/max

			return blkOut;
		}
	}
}
