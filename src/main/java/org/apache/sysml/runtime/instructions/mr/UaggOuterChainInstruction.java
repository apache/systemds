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

package org.apache.sysml.runtime.instructions.mr;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.sysml.lops.PartialAggregate.CorrectionLocationType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.functionobjects.ReduceAll;
import org.apache.sysml.runtime.functionobjects.ReduceCol;
import org.apache.sysml.runtime.functionobjects.ReduceRow;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.LibMatrixOuterAgg;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.DistributedCacheInput;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.mapred.MRBaseForCommonInstructions;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;


/**
 * 
 */
public class UaggOuterChainInstruction extends BinaryInstruction implements IDistributedCacheConsumer
{	
	//operators
	private AggregateUnaryOperator _uaggOp = null;
	private AggregateOperator _aggOp = null;
	private BinaryOperator _bOp = null;
	
	//reused intermediates  
	private MatrixValue _tmpVal1 = null;
	private MatrixValue _tmpVal2 = null;
	
	private double[] _bv = null;
	private int[] _bvi = null;
	
	/**
	 * 
	 * @param bop
	 * @param uaggop
	 * @param in1
	 * @param out
	 * @param istr
	 */
	public UaggOuterChainInstruction(BinaryOperator bop, AggregateUnaryOperator uaggop, AggregateOperator aggop, byte in1, byte in2, byte out, String istr)
	{
		super(null, in1, in2, out, istr);
		
		_uaggOp = uaggop;
		_aggOp = aggop;
		_bOp = bop;
			
		_tmpVal1 = new MatrixBlock();
		_tmpVal2 = new MatrixBlock();
		
		mrtype = MRINSTRUCTION_TYPE.UaggOuterChain;
		instString = istr;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction( String str ) 
		throws DMLRuntimeException 
	{		
		//check number of fields (2/3 inputs, output, type)
		InstructionUtils.checkNumFields ( str, 5 );
		
		//parse instruction parts (without exec type)
		String[] parts = InstructionUtils.getInstructionParts( str );		
		
		AggregateUnaryOperator uaggop = InstructionUtils.parseBasicAggregateUnaryOperator(parts[1]);
		BinaryOperator bop = InstructionUtils.parseBinaryOperator(parts[2]);
		byte in1 = Byte.parseByte(parts[3]);
		byte in2 = Byte.parseByte(parts[4]);
		byte out = Byte.parseByte(parts[5]);
		
		//derive aggregation operator from unary operator
		String aopcode = InstructionUtils.deriveAggregateOperatorOpcode(parts[1]);
		CorrectionLocationType corrLoc = InstructionUtils.deriveAggregateOperatorCorrectionLocation(parts[1]);
		String corrExists = (corrLoc != CorrectionLocationType.NONE) ? "true" : "false";
		AggregateOperator aop = InstructionUtils.parseAggregateOperator(aopcode, corrExists, corrLoc.toString());
	
		return new UaggOuterChainInstruction(bop, uaggop, aop, in1, in2, out, str);
	}
	
	/**
	 * 
	 * @param mcIn
	 * @param mcOut
	 */
	public void computeOutputCharacteristics(MatrixCharacteristics mcIn1, MatrixCharacteristics mcIn2, MatrixCharacteristics mcOut)
	{
		if( _uaggOp.indexFn instanceof ReduceAll )
			mcOut.set(1, 1, mcIn1.getRowsPerBlock(), mcIn2.getColsPerBlock());
		else if( _uaggOp.indexFn instanceof ReduceCol ) //e.g., rowSums
			mcOut.set(mcIn1.getRows(), 1, mcIn1.getRowsPerBlock(), mcIn2.getColsPerBlock());
		else if( _uaggOp.indexFn instanceof ReduceRow ) //e.g., colSums
			mcOut.set(1, mcIn2.getCols(), mcIn1.getRowsPerBlock(), mcIn2.getColsPerBlock());
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass, CachedValueMap cachedValues, 
			           IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		ArrayList<IndexedMatrixValue> blkList = null; 
		boolean rightCached = (_uaggOp.indexFn instanceof ReduceCol || _uaggOp.indexFn instanceof ReduceAll
				               || !LibMatrixOuterAgg.isSupportedUaggOp(_uaggOp, _bOp));
		
		//get the main data input
		if( rightCached ) 
			blkList = cachedValues.get( input1 );
		else // ReduceRow
			blkList = cachedValues.get( input2 );
		
		if( blkList == null )
			return;

		for(IndexedMatrixValue imv : blkList)
		{
			if(imv == null)
				continue;

			MatrixIndexes in1Ix = imv.getIndexes();
			MatrixValue in1Val = imv.getValue();
			
			//allocate space for the intermediate and output value
			IndexedMatrixValue iout = cachedValues.holdPlace(output, valueClass);
			MatrixIndexes outIx = iout.getIndexes();
			MatrixValue outVal = iout.getValue();
			MatrixBlock corr = null;
			
			//get the distributed cache input
			byte dcInputIx = rightCached ? input2 : input1;
			DistributedCacheInput dcInput = MRBaseForCommonInstructions.dcValues.get(dcInputIx);

			//process instruction
			if (LibMatrixOuterAgg.isSupportedUaggOp(_uaggOp, _bOp))
			{
				
				if((LibMatrixOuterAgg.isRowIndexMax(_uaggOp)) || (LibMatrixOuterAgg.isRowIndexMin(_uaggOp))) 
				{
					if( _bv == null ) {
						if( rightCached )
							_bv = dcInput.getRowVectorArray();
						else
							_bv = dcInput.getColumnVectorArray();

						_bvi = LibMatrixOuterAgg.prepareRowIndices(_bv.length, _bv, _bOp, _uaggOp);
					}
				} else {
					//approach: for each ai, do binary search in B, position gives counts
					//step 1: prepare sorted rhs input (once per task)
					if( _bv == null ) {
						if( rightCached )
							_bv = dcInput.getRowVectorArray();
						else
							_bv = dcInput.getColumnVectorArray();
						Arrays.sort(_bv);
					}
				}
		
				LibMatrixOuterAgg.resetOutputMatix(in1Ix, (MatrixBlock)in1Val, outIx, (MatrixBlock)outVal, _uaggOp);
				LibMatrixOuterAgg.aggregateMatrix((MatrixBlock)in1Val, (MatrixBlock)outVal, _bv, _bvi, _bOp, _uaggOp);
			}
			else //default case 
			{
				long in2_cols = dcInput.getNumCols();
				long  in2_colBlocks = (long)Math.ceil(((double)in2_cols)/dcInput.getNumColsPerBlock());
				
				for(int bidx=1; bidx <= in2_colBlocks; bidx++) 
				{
					IndexedMatrixValue imv2 = dcInput.getDataBlock(1, bidx);
					MatrixValue in2Val = imv2.getValue(); 
					
					//outer block operation
					OperationsOnMatrixValues.performBinaryIgnoreIndexes(in1Val, in2Val, _tmpVal1, _bOp);
						
					//unary aggregate operation
					OperationsOnMatrixValues.performAggregateUnary( in1Ix, _tmpVal1, outIx, _tmpVal2, _uaggOp, blockRowFactor, blockColFactor);
					
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
			}
		}
	}
	
	@Override //IDistributedCacheConsumer
	public boolean isDistCacheOnlyIndex( String inst, byte index )
	{
		if(_uaggOp.indexFn instanceof ReduceCol || _uaggOp.indexFn instanceof ReduceAll
			|| !LibMatrixOuterAgg.isSupportedUaggOp(_uaggOp, _bOp)) 
			return (index==input2 && index!=input1);
		else
			return (index==input1 && index!=input2);
	}
	
	@Override //IDistributedCacheConsumer
	public void addDistCacheIndex( String inst, ArrayList<Byte> indexes )
	{
		if(_uaggOp.indexFn instanceof ReduceCol || _uaggOp.indexFn instanceof ReduceAll
			|| !LibMatrixOuterAgg.isSupportedUaggOp(_uaggOp, _bOp)) 
			indexes.add(input2);
		else
			indexes.add(input1);
	}
}
