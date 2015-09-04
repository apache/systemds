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


import java.util.Arrays;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.lops.UAggOuterChain;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.ReduceAll;
import com.ibm.bi.dml.runtime.functionobjects.ReduceCol;
import com.ibm.bi.dml.runtime.functionobjects.ReduceRow;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.mr.BinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.instructions.spark.utils.RDDAggregateUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.LibMatrixOuterAgg;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.util.DataConverter;

/**
 * Two types of broadcast variables used -- 1. Array of type double. 2.PartitionedMatrixBlock
 * 1. Array of type double: Matrix B is sorted at driver level and passed to every task for cases where operations are handled with special cases. e.g. <, RowSum
 * 2. PartitionedMatrixBlock:  Any operations not implemented through this change goes through generic process, In that case, task takes Matrix B, in partitioned form and operate on it.
 */
public class UaggOuterChainSPInstruction extends BinarySPInstruction 
{
	//operators
	private AggregateUnaryOperator _uaggOp = null;
	private AggregateOperator _aggOp = null;
	private BinaryOperator _bOp = null;

	public UaggOuterChainSPInstruction(BinaryOperator bop, AggregateUnaryOperator uaggop, AggregateOperator aggop, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr )
	{
		super(bop, in1, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.UaggOuterChain;
		
		_uaggOp = uaggop;
		_aggOp = aggop;
		_bOp = bop;
			
		_sptype = SPINSTRUCTION_TYPE.UaggOuterChain;
		instString = istr;
	}


	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static UaggOuterChainSPInstruction parseInstruction( String str ) 
		throws DMLRuntimeException 
	{
		String opcode = InstructionUtils.getOpCode(str);

		if ( opcode.equalsIgnoreCase(UAggOuterChain.OPCODE)) {
			String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);

			AggregateUnaryOperator uaggop = InstructionUtils.parseBasicAggregateUnaryOperator(parts[1]);
			BinaryOperator bop = BinaryInstruction.parseBinaryOperator(parts[2]);

			CPOperand in1 = new CPOperand(parts[3]);
			CPOperand in2 = new CPOperand(parts[4]);
			CPOperand out = new CPOperand(parts[5]);
					
			//derive aggregation operator from unary operator
			String aopcode = InstructionUtils.deriveAggregateOperatorOpcode(parts[1]);
			CorrectionLocationType corrLoc = InstructionUtils.deriveAggregateOperatorCorrectionLocation(parts[1]);
			String corrExists = (corrLoc != CorrectionLocationType.NONE) ? "true" : "false";
			AggregateOperator aop = InstructionUtils.parseAggregateOperator(aopcode, corrExists, corrLoc.toString());

			return new UaggOuterChainSPInstruction(bop, uaggop, aop, in1, in2, out, opcode, str);
		} 
		else {
			throw new DMLRuntimeException("UaggOuterChainSPInstruction.parseInstruction():: Unknown opcode " + opcode);
		}

	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{	
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		boolean rightCached = (_uaggOp.indexFn instanceof ReduceCol || _uaggOp.indexFn instanceof ReduceAll
				              || !LibMatrixOuterAgg.isSupportedUaggOp(_uaggOp, _bOp));
		String rddVar = (rightCached) ? input1.getName() : input2.getName();
		String bcastVar = (rightCached) ? input2.getName() : input1.getName();

		//get rdd input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( rddVar );
		MatrixCharacteristics mcIn = sec.getMatrixCharacteristics(rddVar);

		//execute UAggOuterChain instruction
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;		

		if (LibMatrixOuterAgg.isSupportedUaggOp(_uaggOp, _bOp))
		{
			//create sorted broadcast matrix 
			MatrixBlock mb = sec.getMatrixInput(bcastVar);
			sec.releaseMatrixInput(bcastVar);
			bcastVar = null; //prevent lineage tracking
			double[] vmb = DataConverter.convertToDoubleVector(mb);
			Arrays.sort(vmb); 			
			Broadcast<double[]> bv = sec.getSparkContext().broadcast(vmb);
			
			out = in1.mapToPair( new RDDMapUAggOuterChainFunction(bv, _bOp, _uaggOp) );
		}
		else
		{
			Broadcast<PartitionedMatrixBlock> bv = sec.getBroadcastForVariable( bcastVar ); 
			out = in1.mapToPair( new RDDMapGenUAggOuterChainFunction(bv, _uaggOp, _aggOp, _bOp, mcIn.getRowsPerBlock(), mcIn.getColsPerBlock()));	
		}
		
		//final aggregation if required
		if(_uaggOp.indexFn instanceof ReduceAll ) //RC AGG (output is scalar)
		{
			MatrixBlock tmp = RDDAggregateUtils.aggStable(out, _aggOp);
			
			//drop correction after aggregation
			tmp.dropLastRowsOrColums(_aggOp.correctionLocation);

			//put output block into symbol table (no lineage because single block)
			sec.setMatrixOutput(output.getName(), tmp);
		}
		else //R/C AGG (output is rdd)
		{			
			//put output RDD handle into symbol table
			updateUnaryAggOutputMatrixCharacteristics(sec);
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), rddVar);
			if( bcastVar != null )
				sec.addLineageBroadcast(output.getName(), bcastVar);
		}
	}
	
	/**
	 * 
	 * @param sec
	 * @throws DMLRuntimeException
	 */
	protected void updateUnaryAggOutputMatrixCharacteristics(SparkExecutionContext sec) 
		throws DMLRuntimeException
	{	
		String strInput1Name, strInput2Name;
		
		if(_uaggOp.indexFn instanceof ReduceCol) {
			strInput1Name = input1.getName();
			strInput2Name = input2.getName();
		} else {
			strInput1Name = input2.getName();
			strInput2Name = input1.getName();
		}
				
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(strInput1Name);
		MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(strInput2Name);
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
	
		if(!mcOut.dimsKnown()) {
			if(!mc1.dimsKnown()) {
				throw new DMLRuntimeException("The output dimensions are not specified and cannot be inferred from input:" + mc1.toString() + " " + mcOut.toString());
			}
			else {
				//infer statistics from input based on operator
				if( _uaggOp.indexFn instanceof ReduceAll )
					mcOut.set(1, 1, mc1.getRowsPerBlock(), mc1.getColsPerBlock());
				else if (_uaggOp.indexFn instanceof ReduceCol)
					mcOut.set(mc1.getRows(), 1, mc1.getRowsPerBlock(), mc1.getColsPerBlock());
				else if (_uaggOp.indexFn instanceof ReduceRow)
					mcOut.set(1, mc2.getCols(), mc1.getRowsPerBlock(), mc2.getColsPerBlock());
			}
		}
	}
	
	
	/**
	 * 
	 * 
	 */
	private static class RDDMapUAggOuterChainFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 8197406787010296291L;

		private Broadcast<double[]> _bv = null;
		
		private BinaryOperator _bOp = null;
		
		private AggregateUnaryOperator _uaggOp = null;

		//reused intermediates  
		

		
		public RDDMapUAggOuterChainFunction(Broadcast<double[]> bv, BinaryOperator bOp, AggregateUnaryOperator uaggOp)
		{
			// Do not get data from BroadCast variables here, as it will try to deserialize the data whenever it gets instantiated through driver class. This will cause unnecessary delay in iinstantiating class
			// through driver, and overall process.
			// Instead of this let task gets data from BroadCast variable whenever required, as data is already available in memory for task to fetch.

			//Sorted array
			_bv = bv;
			_bOp = bOp;
			_uaggOp = uaggOp;
			
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			MatrixIndexes in1Ix = arg0._1();
			MatrixBlock in1Val  = arg0._2();

			MatrixIndexes outIx = new MatrixIndexes();
			MatrixBlock outVal = new MatrixBlock();
			
			LibMatrixOuterAgg.aggregateMatrix(in1Ix, in1Val, outIx, outVal, _bv.value(), _bOp, _uaggOp);

			return new Tuple2<MatrixIndexes, MatrixBlock>(outIx, outVal);
		}
	}
	
	private static class RDDMapGenUAggOuterChainFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 8197406787010296291L;

		private Broadcast<PartitionedMatrixBlock> _pbc = null;
		
		// Operators
		private AggregateUnaryOperator _uaggOp = null;
		private AggregateOperator _aggOp = null;
		private BinaryOperator _bOp = null;

		private int _brlen, _bclen;
		
		//reused intermediates  
		private MatrixValue _tmpVal1 = null;
		private MatrixValue _tmpVal2 = null;
		

		
		public RDDMapGenUAggOuterChainFunction(Broadcast<PartitionedMatrixBlock> binput, AggregateUnaryOperator uaggOp, AggregateOperator aggOp, BinaryOperator bOp, 
				int brlen, int bclen)
		{
			//partition vector for fast in memory lookup
			_pbc = binput;
			
			// Operators
			_uaggOp = uaggOp;
			_aggOp = aggOp;
			_bOp = bOp;
			
			//Matrix dimension (row, column)
			_brlen = brlen;
			_bclen = bclen;
			
			_tmpVal1 = new MatrixBlock();
			_tmpVal2 = new MatrixBlock();
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			PartitionedMatrixBlock pm = _pbc.value();
			
			MatrixIndexes in1Ix = arg0._1();
			MatrixBlock in1Val  = arg0._2();

			MatrixIndexes outIx = new MatrixIndexes();
			MatrixBlock outVal = new MatrixBlock();
			
			MatrixBlock corr = null;
			
				
			long  in2_colBlocks = pm.getNumColumnBlocks();
			
			for(int bidx=1; bidx <= in2_colBlocks; bidx++) 
			{
				MatrixValue in2Val = pm.getMatrixBlock(1, bidx);
				
				//outer block operation
				OperationsOnMatrixValues.performBinaryIgnoreIndexes(in1Val, in2Val, _tmpVal1, _bOp);
					
				//unary aggregate operation
				OperationsOnMatrixValues.performAggregateUnary( in1Ix, _tmpVal1, outIx, _tmpVal2, _uaggOp, _brlen, _bclen);
				
				//aggregate over all rhs blocks
				if( corr == null ) {
					outVal.reset(_tmpVal2.getNumRows(), _tmpVal2.getNumColumns(), false);
					corr = new MatrixBlock(_tmpVal2.getNumRows(), _tmpVal2.getNumColumns(), false);
				}
				
				if(_aggOp.correctionExists)
					OperationsOnMatrixValues.incrementalAggregation(outVal, corr, _tmpVal2, _aggOp, true);
				else 
					OperationsOnMatrixValues.incrementalAggregation(outVal, null, _tmpVal2, _aggOp, true);
			}
			
			return new Tuple2<MatrixIndexes, MatrixBlock>(outIx, outVal);
		}
	}

}
