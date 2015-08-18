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
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;


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
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		InstructionUtils.checkNumFields ( str, 4 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType( str );
		
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		out.split(parts[3]);
		double init = Double.parseDouble(parts[4]);
		
		return new CumulativeOffsetSPInstruction(null, in1, in2, out, init, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
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
				      .mapToPair(new RDDCumOffsetFunction(_uop, _bop));
		
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
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
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
			
			return ret;
		}
	}
	
	/**
	 * 
	 */
	private static class RDDCumOffsetFunction implements PairFunction<Tuple2<MatrixIndexes,Tuple2<MatrixBlock, MatrixBlock>>, MatrixIndexes, MatrixBlock> 
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
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>> arg0)
			throws Exception 
		{
			//prepare inputs and outputs
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock dblkIn = arg0._2()._1(); //original data 
			MatrixBlock oblkIn = arg0._2()._2(); //offset row vector
			MatrixBlock blkOut = new MatrixBlock(dblkIn.getNumRows(), dblkIn.getNumColumns(), dblkIn.isInSparseFormat());
			
			//blockwise offset aggregation and prefix sum computation
			MatrixBlock data2 = new MatrixBlock(dblkIn); //cp data
			MatrixBlock fdata2 = data2.sliceOperations(0, 0, 0, data2.getNumColumns()-1, new MatrixBlock()); //1-based
			fdata2.binaryOperationsInPlace(_bop, oblkIn); //sum offset to first row
			data2.copy(0, 0, 0, data2.getNumColumns()-1, fdata2, true); //0-based
			data2.unaryOperations(_uop, blkOut); //compute columnwise prefix sums/prod/min/max

			//set output indexes
			MatrixIndexes ixOut = new MatrixIndexes(ixIn);
			return new Tuple2<MatrixIndexes,MatrixBlock>(ixOut, blkOut);
		}
	}
}
