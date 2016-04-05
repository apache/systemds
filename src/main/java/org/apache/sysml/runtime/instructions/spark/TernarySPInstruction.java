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

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import org.apache.sysml.lops.Ternary;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.functionobjects.CTable;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.CTableMap;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.SimpleOperator;
import org.apache.sysml.runtime.util.LongLongDoubleHashMap.LLDoubleEntry;
import org.apache.sysml.runtime.util.UtilFunctions;

public class TernarySPInstruction extends ComputationSPInstruction
{
	
	private String _outDim1;
	private String _outDim2;
	private boolean _dim1Literal; 
	private boolean _dim2Literal;
	private boolean _isExpand;
	private boolean _ignoreZeros;
	
	public TernarySPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, 
							 String outputDim1, boolean dim1Literal,String outputDim2, boolean dim2Literal, 
							 boolean isExpand, boolean ignoreZeros, String opcode, String istr )
	{
		super(op, in1, in2, in3, out, opcode, istr);
		_outDim1 = outputDim1;
		_dim1Literal = dim1Literal;
		_outDim2 = outputDim2;
		_dim2Literal = dim2Literal;
		_isExpand = isExpand;
		_ignoreZeros = ignoreZeros;
	}

	/**
	 * 
	 * @param inst
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static TernarySPInstruction parseInstruction(String inst) 
		throws DMLRuntimeException
	{	
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(inst);
		InstructionUtils.checkNumFields ( parts, 7 );
		
		String opcode = parts[0];
		
		//handle opcode
		if ( !(opcode.equalsIgnoreCase("ctable") || opcode.equalsIgnoreCase("ctableexpand")) ) {
			throw new DMLRuntimeException("Unexpected opcode in TertiarySPInstruction: " + inst);
		}
		boolean isExpand = opcode.equalsIgnoreCase("ctableexpand");
		
		//handle operands
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		
		//handle known dimension information
		String[] dim1Fields = parts[4].split(Instruction.LITERAL_PREFIX);
		String[] dim2Fields = parts[5].split(Instruction.LITERAL_PREFIX);

		CPOperand out = new CPOperand(parts[6]);
		boolean ignoreZeros = Boolean.parseBoolean(parts[7]);
		
		// ctable does not require any operator, so we simply pass-in a dummy operator with null functionobject
		return new TernarySPInstruction(new SimpleOperator(null), in1, in2, in3, out, dim1Fields[0], Boolean.parseBoolean(dim1Fields[1]), dim2Fields[0], Boolean.parseBoolean(dim2Fields[1]), isExpand, ignoreZeros, opcode, inst);
	}


	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException 
	{	
		SparkExecutionContext sec = (SparkExecutionContext)ec;
	
		//get input rdd handle
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = null;
		JavaPairRDD<MatrixIndexes,MatrixBlock> in3 = null;
		double scalar_input2 = -1, scalar_input3 = -1;
		
		Ternary.OperationTypes ctableOp = Ternary.findCtableOperationByInputDataTypes(
				input1.getDataType(), input2.getDataType(), input3.getDataType());
		ctableOp = _isExpand ? Ternary.OperationTypes.CTABLE_EXPAND_SCALAR_WEIGHT : ctableOp;
		
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		
		// First get the block sizes and then set them as -1 to allow for binary cell reblock
		int brlen = mc1.getRowsPerBlock();
		int bclen = mc1.getColsPerBlock();
		
		JavaPairRDD<MatrixIndexes, ArrayList<MatrixBlock>> inputMBs = null;
		JavaPairRDD<MatrixIndexes, CTableMap> ctables = null;
		JavaPairRDD<MatrixIndexes, Double> bincellsNoFilter = null;
		boolean setLineage2 = false;
		boolean setLineage3 = false;
		switch(ctableOp) {
			case CTABLE_TRANSFORM: //(VECTOR)
				// F=ctable(A,B,W) 
				in2 = sec.getBinaryBlockRDDHandleForVariable( input2.getName() );
				in3 = sec.getBinaryBlockRDDHandleForVariable( input3.getName() );
				setLineage2 = true;
				setLineage3 = true;
				
				inputMBs = in1.cogroup(in2).cogroup(in3)
							.mapToPair(new MapThreeMBIterableIntoAL());
				
				ctables = inputMBs.mapToPair(new PerformCTableMapSideOperation(ctableOp, scalar_input2, 
							scalar_input3, this.instString, (SimpleOperator)_optr, _ignoreZeros));
				break;
			
				
			case CTABLE_EXPAND_SCALAR_WEIGHT: //(VECTOR)
				// F = ctable(seq,A) or F = ctable(seq,B,1)
				scalar_input3 = sec.getScalarInput(input3.getName(), input3.getValueType(), input3.isLiteral()).getDoubleValue();
				if(scalar_input3 == 1) {
					in2 = sec.getBinaryBlockRDDHandleForVariable( input2.getName() );
					setLineage2 = true;
					bincellsNoFilter = in2.flatMapToPair(new ExpandScalarCtableOperation(brlen));
					break;
				}
			case CTABLE_TRANSFORM_SCALAR_WEIGHT: //(VECTOR/MATRIX)
				// F = ctable(A,B) or F = ctable(A,B,1)
				in2 = sec.getBinaryBlockRDDHandleForVariable( input2.getName() );
				setLineage2 = true;

				scalar_input3 = sec.getScalarInput(input3.getName(), input3.getValueType(), input3.isLiteral()).getDoubleValue();
				inputMBs = in1.cogroup(in2).mapToPair(new MapTwoMBIterableIntoAL());
				
				ctables = inputMBs.mapToPair(new PerformCTableMapSideOperation(ctableOp, scalar_input2, 
						scalar_input3, this.instString, (SimpleOperator)_optr, _ignoreZeros));
				break;
				
			case CTABLE_TRANSFORM_HISTOGRAM: //(VECTOR)
				// F=ctable(A,1) or F = ctable(A,1,1)
				scalar_input2 = sec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral()).getDoubleValue();
				scalar_input3 = sec.getScalarInput(input3.getName(), input3.getValueType(), input3.isLiteral()).getDoubleValue();
				inputMBs = in1.mapToPair(new MapMBIntoAL());
				
				ctables = inputMBs.mapToPair(new PerformCTableMapSideOperation(ctableOp, scalar_input2, 
						scalar_input3, this.instString, (SimpleOperator)_optr, _ignoreZeros));
				break;
				
			case CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM: //(VECTOR)
				// F=ctable(A,1,W)
				in3 = sec.getBinaryBlockRDDHandleForVariable( input3.getName() );
				setLineage3 = true;
				
				scalar_input2 = sec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral()).getDoubleValue();
				inputMBs = in1.cogroup(in3).mapToPair(new MapTwoMBIterableIntoAL());
				
				ctables = inputMBs.mapToPair(new PerformCTableMapSideOperation(ctableOp, scalar_input2, 
						scalar_input3, this.instString, (SimpleOperator)_optr, _ignoreZeros));
				break;
			
			default:
				throw new DMLRuntimeException("Encountered an invalid ctable operation ("+ctableOp+") while executing instruction: " + this.toString());
		}
		
		// Now perform aggregation on ctables to get binaryCells 
		if(bincellsNoFilter == null && ctables != null) {
			bincellsNoFilter =  
					ctables.values()
					.flatMapToPair(new ExtractBinaryCellsFromCTable());
			bincellsNoFilter = RDDAggregateUtils.sumCellsByKeyStable(bincellsNoFilter);
		}
		else if(!(bincellsNoFilter != null && ctables == null)) {
			throw new DMLRuntimeException("Incorrect ctable operation");
		}
		
		// handle known/unknown dimensions
		long outputDim1 = (_dim1Literal ? (long) Double.parseDouble(_outDim1) : (sec.getScalarInput(_outDim1, ValueType.DOUBLE, false)).getLongValue());
		long outputDim2 = (_dim2Literal ? (long) Double.parseDouble(_outDim2) : (sec.getScalarInput(_outDim2, ValueType.DOUBLE, false)).getLongValue());
		MatrixCharacteristics mcBinaryCells = null;
		boolean findDimensions = (outputDim1 == -1 && outputDim2 == -1); 
		
		if( !findDimensions ) {
			if((outputDim1 == -1 && outputDim2 != -1) || (outputDim1 != -1 && outputDim2 == -1))
				throw new DMLRuntimeException("Incorrect output dimensions passed to TernarySPInstruction:" + outputDim1 + " " + outputDim2);
			else 
				mcBinaryCells = new MatrixCharacteristics(outputDim1, outputDim2, brlen, bclen);	
			
			// filtering according to given dimensions
			bincellsNoFilter = bincellsNoFilter
					.filter(new FilterCells(mcBinaryCells.getRows(), mcBinaryCells.getCols()));
		}
		
		// convert double values to matrix cell
		JavaPairRDD<MatrixIndexes, MatrixCell> binaryCells = bincellsNoFilter
				.mapToPair(new ConvertToBinaryCell());
		
		// find dimensions if necessary (w/ cache for reblock)
		if( findDimensions ) {						
			binaryCells = SparkUtils.cacheBinaryCellRDD(binaryCells);
			mcBinaryCells = SparkUtils.computeMatrixCharacteristics(binaryCells);
		}
		
		//store output rdd handle
		sec.setRDDHandleForVariable(output.getName(), binaryCells);
		mcOut.set(mcBinaryCells);
		// Since we are outputing binary cells, we set block sizes = -1
		mcOut.setRowsPerBlock(-1); mcOut.setColsPerBlock(-1);
		sec.addLineageRDD(output.getName(), input1.getName());
		if(setLineage2)
			sec.addLineageRDD(output.getName(), input2.getName());
		if(setLineage3)
			sec.addLineageRDD(output.getName(), input3.getName());
	}	
	
	/**
	 *
	 */
	private static class ExpandScalarCtableOperation implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, Double> 
	{
		private static final long serialVersionUID = -12552669148928288L;
	
		private int _brlen;
		
		public ExpandScalarCtableOperation(int brlen) {
			_brlen = brlen;
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, Double>> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
			throws Exception 
		{
			MatrixIndexes ix = arg0._1();
			MatrixBlock mb = arg0._2(); //col-vector
			
			//create an output cell per matrix block row (aligned w/ original source position)
			ArrayList<Tuple2<MatrixIndexes, Double>> retVal = new ArrayList<Tuple2<MatrixIndexes,Double>>();
			CTable ctab = CTable.getCTableFnObject();
			for( int i=0; i<mb.getNumRows(); i++ )
			{
				//compute global target indexes (via ctable obj for error handling consistency)
				long row = UtilFunctions.computeCellIndex(ix.getRowIndex(), _brlen, i);
				double v2 = mb.quickGetValue(i, 0);
				Pair<MatrixIndexes,Double> p = ctab.execute(row, v2, 1.0);
				
				//indirect construction over pair to avoid tuple2 dependency in general ctable obj
				if( p.getKey().getRowIndex() >= 1 ) //filter rejected entries
					retVal.add(new Tuple2<MatrixIndexes,Double>(p.getKey(), p.getValue()));
			}
			
			return retVal;
		}
	}
	
	private static class MapTwoMBIterableIntoAL implements PairFunction<Tuple2<MatrixIndexes,Tuple2<Iterable<MatrixBlock>,Iterable<MatrixBlock>>>, MatrixIndexes, ArrayList<MatrixBlock>> {

		private static final long serialVersionUID = 271459913267735850L;

		private MatrixBlock extractBlock(Iterable<MatrixBlock> blks, MatrixBlock retVal) throws Exception {
			for(MatrixBlock blk1 : blks) {
				if(retVal != null) {
					throw new Exception("ERROR: More than 1 matrixblock found for one of the inputs at a given index");
				}
				retVal = blk1;
			}
			if(retVal == null) {
				throw new Exception("ERROR: No matrixblock found for one of the inputs at a given index");
			}
			return retVal;
		}
		
		@Override
		public Tuple2<MatrixIndexes, ArrayList<MatrixBlock>> call(
				Tuple2<MatrixIndexes, Tuple2<Iterable<MatrixBlock>, Iterable<MatrixBlock>>> kv)
				throws Exception {
			MatrixBlock in1 = null; MatrixBlock in2 = null;
			in1 = extractBlock(kv._2._1, in1);
			in2 = extractBlock(kv._2._2, in2);
			// Now return unflatten AL
			ArrayList<MatrixBlock> inputs = new ArrayList<MatrixBlock>();
			inputs.add(in1); inputs.add(in2);  
			return new Tuple2<MatrixIndexes, ArrayList<MatrixBlock>>(kv._1, inputs);
		}
		
	}
	
	private static class MapThreeMBIterableIntoAL implements PairFunction<Tuple2<MatrixIndexes,Tuple2<Iterable<Tuple2<Iterable<MatrixBlock>,Iterable<MatrixBlock>>>,Iterable<MatrixBlock>>>, MatrixIndexes, ArrayList<MatrixBlock>> {

		private static final long serialVersionUID = -4873754507037646974L;
		
		private MatrixBlock extractBlock(Iterable<MatrixBlock> blks, MatrixBlock retVal) throws Exception {
			for(MatrixBlock blk1 : blks) {
				if(retVal != null) {
					throw new Exception("ERROR: More than 1 matrixblock found for one of the inputs at a given index");
				}
				retVal = blk1;
			}
			if(retVal == null) {
				throw new Exception("ERROR: No matrixblock found for one of the inputs at a given index");
			}
			return retVal;
		}

		@Override
		public Tuple2<MatrixIndexes, ArrayList<MatrixBlock>> call(
				Tuple2<MatrixIndexes, Tuple2<Iterable<Tuple2<Iterable<MatrixBlock>, Iterable<MatrixBlock>>>, Iterable<MatrixBlock>>> kv)
				throws Exception {
			MatrixBlock in1 = null; MatrixBlock in2 = null; MatrixBlock in3 = null;
			
			for(Tuple2<Iterable<MatrixBlock>, Iterable<MatrixBlock>> blks : kv._2._1) {
				in1 = extractBlock(blks._1, in1);
				in2 = extractBlock(blks._2, in2);
			}
			in3 = extractBlock(kv._2._2, in3);
			
			// Now return unflatten AL
			ArrayList<MatrixBlock> inputs = new ArrayList<MatrixBlock>();
			inputs.add(in1); inputs.add(in2); inputs.add(in3);  
			return new Tuple2<MatrixIndexes, ArrayList<MatrixBlock>>(kv._1, inputs);
		}
		
	}
	
	private static class PerformCTableMapSideOperation implements PairFunction<Tuple2<MatrixIndexes,ArrayList<MatrixBlock>>, MatrixIndexes, CTableMap> {

		private static final long serialVersionUID = 5348127596473232337L;

		Ternary.OperationTypes ctableOp;
		double scalar_input2; double scalar_input3;
		String instString;
		Operator optr;
		boolean ignoreZeros;
		
		public PerformCTableMapSideOperation(Ternary.OperationTypes ctableOp, double scalar_input2, double scalar_input3, String instString, Operator optr, boolean ignoreZeros) {
			this.ctableOp = ctableOp;
			this.scalar_input2 = scalar_input2;
			this.scalar_input3 = scalar_input3;
			this.instString = instString;
			this.optr = optr;
			this.ignoreZeros = ignoreZeros;
		}
		
		private void expectedALSize(int length, ArrayList<MatrixBlock> al) throws Exception {
			if(al.size() != length) {
				throw new Exception("Expected arraylist of size:" + length + ", but found " + al.size());
			}
		}
		
		@Override
		public Tuple2<MatrixIndexes, CTableMap> call(
				Tuple2<MatrixIndexes, ArrayList<MatrixBlock>> kv) throws Exception {
			CTableMap ctableResult = new CTableMap();
			MatrixBlock ctableResultBlock = null;
			
			IndexedMatrixValue in1, in2, in3 = null;
			in1 = new IndexedMatrixValue(kv._1, kv._2.get(0));
			MatrixBlock matBlock1 = kv._2.get(0);
			
			switch( ctableOp )
			{
				case CTABLE_TRANSFORM: {
					in2 = new IndexedMatrixValue(kv._1, kv._2.get(1));
					in3 = new IndexedMatrixValue(kv._1, kv._2.get(2));
					expectedALSize(3, kv._2);
					
					if(in1==null || in2==null || in3 == null )
						break;	
					else
						OperationsOnMatrixValues.performTernary(in1.getIndexes(), in1.getValue(), in2.getIndexes(), in2.getValue(), 
                                in3.getIndexes(), in3.getValue(), ctableResult, ctableResultBlock, optr);
					break;
				}
				case CTABLE_TRANSFORM_SCALAR_WEIGHT: 
				case CTABLE_EXPAND_SCALAR_WEIGHT:
				{
					// 3rd input is a scalar
					in2 = new IndexedMatrixValue(kv._1, kv._2.get(1));
					expectedALSize(2, kv._2);
					if(in1==null || in2==null )
						break;
					else
						matBlock1.ternaryOperations((SimpleOperator)optr, kv._2.get(1), scalar_input3, ignoreZeros, ctableResult, ctableResultBlock);
						break;
				}
				case CTABLE_TRANSFORM_HISTOGRAM: {
					expectedALSize(1, kv._2);
					OperationsOnMatrixValues.performTernary(in1.getIndexes(), in1.getValue(), scalar_input2, 
							scalar_input3, ctableResult, ctableResultBlock, optr);
					break;
				}
				case CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM: {
					// 2nd and 3rd inputs are scalars
					expectedALSize(2, kv._2);
					in3 = new IndexedMatrixValue(kv._1, kv._2.get(1)); // Note: kv._2.get(1), not kv._2.get(2)
					
					if(in1==null || in3==null)
						break;
					else
						OperationsOnMatrixValues.performTernary(in1.getIndexes(), in1.getValue(), scalar_input2, 
								in3.getIndexes(), in3.getValue(), ctableResult, ctableResultBlock, optr);		
					break;
				}
				default:
					throw new DMLRuntimeException("Unrecognized opcode in Tertiary Instruction: " + instString);		
			}
			
			return new Tuple2<MatrixIndexes, CTableMap>(kv._1, ctableResult);
		}
		
	}
	
	private static class MapMBIntoAL implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, ArrayList<MatrixBlock>> {

		private static final long serialVersionUID = 2068398913653350125L;

		@Override
		public Tuple2<MatrixIndexes, ArrayList<MatrixBlock>> call(
				Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
			ArrayList<MatrixBlock> retVal = new ArrayList<MatrixBlock>();
			retVal.add(kv._2);
			return new Tuple2<MatrixIndexes, ArrayList<MatrixBlock>>(kv._1, retVal);
		}
		
	}
	
	private static class ExtractBinaryCellsFromCTable implements PairFlatMapFunction<CTableMap, MatrixIndexes, Double> {

		private static final long serialVersionUID = -5933677686766674444L;
		
		@SuppressWarnings("deprecation")
		@Override
		public Iterable<Tuple2<MatrixIndexes, Double>> call(CTableMap ctableMap)
				throws Exception {
			ArrayList<Tuple2<MatrixIndexes, Double>> retVal = new ArrayList<Tuple2<MatrixIndexes, Double>>();
			
			for(LLDoubleEntry ijv : ctableMap.entrySet()) {
				long i = ijv.key1;
				long j =  ijv.key2;
				double v =  ijv.value;
				
				// retVal.add(new Tuple2<MatrixIndexes, MatrixCell>(blockIndexes, cell));
				retVal.add(new Tuple2<MatrixIndexes, Double>(new MatrixIndexes(i, j), v));
			}
			return retVal;
		}
		
	}
	
	private static class ConvertToBinaryCell implements PairFunction<Tuple2<MatrixIndexes,Double>, MatrixIndexes, MatrixCell> {

		private static final long serialVersionUID = 7481186480851982800L;
		
		@Override
		public Tuple2<MatrixIndexes, MatrixCell> call(
				Tuple2<MatrixIndexes, Double> kv) throws Exception {
			
			MatrixCell cell = new MatrixCell(kv._2().doubleValue());
			return new Tuple2<MatrixIndexes, MatrixCell>(kv._1(), cell);
		}
		
	}
	
	private static class FilterCells implements Function<Tuple2<MatrixIndexes,Double>, Boolean> {
		private static final long serialVersionUID = 108448577697623247L;

		long rlen; long clen;
		public FilterCells(long rlen, long clen) {
			this.rlen = rlen;
			this.clen = clen;
		}
		
		@Override
		public Boolean call(Tuple2<MatrixIndexes, Double> kv) throws Exception {
			if(kv._1.getRowIndex() <= 0 || kv._1.getColumnIndex() <= 0) {
				throw new Exception("Incorrect cell values in TernarySPInstruction:" + kv._1);
			}
			if(kv._1.getRowIndex() <= rlen && kv._1.getColumnIndex() <= clen) {
				return true;
			}
			return false;
		}
		
	}
}
