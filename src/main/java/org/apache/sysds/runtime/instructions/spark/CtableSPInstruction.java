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

import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.lops.Ctable;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.ReblockBuffer;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysds.runtime.matrix.data.CTableMap;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.LongLongDoubleHashMap.ADoubleEntry;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;

public class CtableSPInstruction extends ComputationSPInstruction {
	private String _outDim1;
	private String _outDim2;
	private boolean _dim1Literal;
	private boolean _dim2Literal;
	private boolean _isExpand;
	private final boolean _ignoreZeros;
	private final boolean _outputEmptyBlocks;

	private CtableSPInstruction(CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
			String outputDim1, boolean dim1Literal, String outputDim2, boolean dim2Literal, boolean isExpand,
			boolean ignoreZeros, boolean outputEmptyBlocks, String opcode, String istr) {
		super(SPType.Ctable, null, in1, in2, in3, out, opcode, istr);
		_outDim1 = outputDim1;
		_dim1Literal = dim1Literal;
		_outDim2 = outputDim2;
		_dim2Literal = dim2Literal;
		_isExpand = isExpand;
		_ignoreZeros = ignoreZeros;
		_outputEmptyBlocks = outputEmptyBlocks;
	}

	public static CtableSPInstruction parseInstruction(String inst) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(inst);
		InstructionUtils.checkNumFields ( parts, 8 );
		
		String opcode = parts[0];
		
		//handle opcode
		if ( !(opcode.equalsIgnoreCase(Opcodes.CTABLE.toString()) || opcode.equalsIgnoreCase(Opcodes.CTABLEEXPAND.toString())) ) {
			throw new DMLRuntimeException("Unexpected opcode in TertiarySPInstruction: " + inst);
		}
		boolean isExpand = opcode.equalsIgnoreCase(Opcodes.CTABLEEXPAND.toString());
		
		//handle operands
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		
		//handle known dimension information
		String[] dim1Fields = parts[4].split(Instruction.LITERAL_PREFIX);
		String[] dim2Fields = parts[5].split(Instruction.LITERAL_PREFIX);

		CPOperand out = new CPOperand(parts[6]);
		boolean ignoreZeros = Boolean.parseBoolean(parts[7]);
		boolean outputEmptyBlocks = Boolean.parseBoolean(parts[8]);
		
		// ctable does not require any operator, so we simply pass-in a dummy operator with null functionobject
		return new CtableSPInstruction(in1, in2, in3, out, dim1Fields[0], Boolean.parseBoolean(dim1Fields[1]),
			dim2Fields[0], Boolean.parseBoolean(dim2Fields[1]), isExpand, ignoreZeros, outputEmptyBlocks, opcode, inst);
	}


	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
	
		Ctable.OperationTypes ctableOp = Ctable.findCtableOperationByInputDataTypes(
			input1.getDataType(), input2.getDataType(), input3.getDataType());
		ctableOp = _isExpand ? Ctable.OperationTypes.CTABLE_EXPAND_SCALAR_WEIGHT : ctableOp;
		
		//get input rdd handle
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = !ctableOp.hasSecondInput() ? null :
			sec.getBinaryMatrixBlockRDDHandleForVariable( input2.getName() );
		
		JavaPairRDD<MatrixIndexes,MatrixBlock> in3 = null;
		double s2 = -1, s3 = -1; //scalars
		
		DataCharacteristics mc1 = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
		
		// handle known/unknown dimensions
		long dim1 = (_dim1Literal ? (long) Double.parseDouble(_outDim1) :
			(sec.getScalarInput(_outDim1, ValueType.FP64, false)).getLongValue());
		long dim2 = (_dim2Literal ? (long) Double.parseDouble(_outDim2) :
			(sec.getScalarInput(_outDim2, ValueType.FP64, false)).getLongValue());
		if( dim1 == -1 && dim2 == -1 ) {
			//note: if we need to determine the dimensions to we do so before 
			//creating cells to avoid unnecessary caching, repeated joins, etc.
			dim1 = (long) RDDAggregateUtils.max(in1);
			dim2 = ctableOp.hasSecondInput() ? (long) RDDAggregateUtils.max(in2) :
				sec.getScalarInput(input3).getLongValue();
		}
		mcOut.set(dim1, dim2, mc1.getBlocksize());
		mcOut.setNonZerosBound(mc1.getLength()); //vector or matrix
		mcOut.setNoEmptyBlocks(!_outputEmptyBlocks);
		if( !mcOut.dimsKnown() )
			throw new DMLRuntimeException("Unknown ctable output dimensions: "+mcOut);
		
		//compute preferred degree of parallelism
		int numParts = Math.max(4 * (mc1.dimsKnown() ?
			SparkUtils.getNumPreferredPartitions(mc1) : in1.getNumPartitions()),
			SparkUtils.getNumPreferredPartitions(mcOut, _outputEmptyBlocks));
		
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = null;
		switch(ctableOp) {
			case CTABLE_TRANSFORM: //(VECTOR)
				// F=ctable(A,B,W) 
				in3 = sec.getBinaryMatrixBlockRDDHandleForVariable( input3.getName() );
				out = in1.join(in2, numParts).join(in3, numParts)
					.mapValues(new MapJoinSignature3())
					.mapPartitionsToPair(new CTableFunction(ctableOp, s2, s3, _ignoreZeros, mcOut));
				break;
			
			case CTABLE_EXPAND_SCALAR_WEIGHT: //(VECTOR)
			case CTABLE_TRANSFORM_SCALAR_WEIGHT: //(VECTOR/MATRIX)
				// F = ctable(A,B) or F = ctable(A,B,1)
				s3 = sec.getScalarInput(input3).getDoubleValue();
				out = in1.join(in2, numParts).mapValues(new MapJoinSignature2())
					.mapPartitionsToPair(new CTableFunction(ctableOp, s2, s3, _ignoreZeros, mcOut));
				break;
				
			case CTABLE_TRANSFORM_HISTOGRAM: //(VECTOR)
				// F=ctable(A,1) or F = ctable(A,1,1)
				s2 = sec.getScalarInput(input2).getDoubleValue();
				s3 = sec.getScalarInput(input3).getDoubleValue();
				out = in1.mapValues(new MapJoinSignature1())
					.mapPartitionsToPair(new CTableFunction(ctableOp, s2, s3, _ignoreZeros, mcOut));
				break;
				
			case CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM: //(VECTOR)
				// F=ctable(A,1,W)
				in3 = sec.getBinaryMatrixBlockRDDHandleForVariable( input3.getName() );
				s2 = sec.getScalarInput(input2).getDoubleValue();
				out = in1.join(in3, numParts).mapValues(new MapJoinSignature2())
					.mapPartitionsToPair(new CTableFunction(ctableOp, s2, s3, _ignoreZeros, mcOut));
				break;
			
			default:
				throw new DMLRuntimeException("Encountered an invalid ctable operation ("+ctableOp+") while executing instruction: " + this.toString());
		}
		
		//perform fused aggregation and reblock
		out = !_outputEmptyBlocks ? out :
			out.union(SparkUtils.getEmptyBlockRDD(sec.getSparkContext(), mcOut));
		out = RDDAggregateUtils.sumByKeyStable(out, numParts, false);
		
		//store output rdd handle
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		if( ctableOp.hasSecondInput() )
			sec.addLineageRDD(output.getName(), input2.getName());
		if( ctableOp.hasThirdInput() )
			sec.addLineageRDD(output.getName(), input3.getName());
		
		//post-processing to obtain sparsity of ultra-sparse outputs
		SparkUtils.postprocessUltraSparseOutput(sec.getMatrixObject(output), mcOut);
	}

	public CPOperand getOutDim1() {
		return new CPOperand(_outDim1, ValueType.FP64, Types.DataType.SCALAR, _dim1Literal);
	}

	public CPOperand getOutDim2() {
		return new CPOperand(_outDim1, ValueType.FP64, Types.DataType.SCALAR, _dim1Literal);
	}

	public boolean getIsExpand() {
		return _isExpand;
	}

	public boolean getIgnoreZeros() {
		return _ignoreZeros;
	}

	private static class CTableFunction implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes, MatrixBlock[]>>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 5348127596473232337L;

		private final Ctable.OperationTypes _ctableOp;
		private final double _scalar_input2, _scalar_input3;
		private final boolean _ignoreZeros;
		private final long _dim1, _dim2;
		private final int _blen;
		
		public CTableFunction(Ctable.OperationTypes ctableOp, double s2, double s3, boolean ignoreZeros, DataCharacteristics mcOut) {
			this(ctableOp, s2, s3, ignoreZeros, false, mcOut);
		}
		
		public CTableFunction(Ctable.OperationTypes ctableOp, double s2, double s3, boolean ignoreZeros, boolean emitEmpty, DataCharacteristics mcOut) {
			_ctableOp = ctableOp;
			_scalar_input2 = s2;
			_scalar_input3 = s3;
			_ignoreZeros = ignoreZeros;
			_dim1 = mcOut.getRows();
			_dim2 = mcOut.getCols();
			_blen = mcOut.getBlocksize();
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixIndexes, MatrixBlock[]>> arg0)
			throws Exception
		{
			CTableMap map = new CTableMap(); MatrixBlock block = null;
			
			//local aggregation of entire partition
			while( arg0.hasNext() ) {
				Tuple2<MatrixIndexes,MatrixBlock[]> tmp = arg0.next();
				MatrixBlock[] mb = tmp._2();
				
				switch( _ctableOp ) {
					case CTABLE_TRANSFORM: {
						mb[0].ctableOperations(null, mb[1], mb[2], map, block);
						break;
					}
					case CTABLE_EXPAND_SCALAR_WEIGHT:
					case CTABLE_TRANSFORM_SCALAR_WEIGHT: {
						// 3rd input is a scalar
						mb[0].ctableOperations(null, mb[1], _scalar_input3, _ignoreZeros, map, block);
						break;
					}
					case CTABLE_TRANSFORM_HISTOGRAM: {
						mb[0].ctableOperations(null, _scalar_input2, _scalar_input3, map, block);
						break;
					}
					case CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM: {
						// 2nd and 3rd inputs are scalars
						mb[0].ctableOperations(null, _scalar_input2, mb[1], map, block);
						break;
					}
					default:
						break;
				}
			}
			
			ReblockBuffer rbuff = new ReblockBuffer(Math.min(
				4*1024*1024, map.size()), _dim1, _dim2, _blen);
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<>();
			
			//append to buffer for blocked output
			Iterator<ADoubleEntry> iter = map.getIterator();
			while( iter.hasNext() ) {
				ADoubleEntry e = iter.next();
				if( e.getKey1() <= _dim1 && e.getKey2() <= _dim2 ) { 
					if( rbuff.getSize() >= rbuff.getCapacity() )
						flushBufferToList(rbuff, ret);
					rbuff.appendCell(e.getKey1(), e.getKey2(), e.value);
				}
			}
			
			//final flush buffer
			if( rbuff.getSize() > 0 )
				flushBufferToList(rbuff, ret);
			
			return ret.iterator();
		}
	
		protected void flushBufferToList( ReblockBuffer rbuff,  ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret ) 
			throws DMLRuntimeException
		{
			rbuff.flushBufferToBinaryBlocks().stream() // prevent library dependencies
				.map(b -> SparkUtils.fromIndexedMatrixBlock(b)).forEach(b -> ret.add(b));
		}
	}
	
	public static class MapJoinSignature1 implements Function<MatrixBlock, MatrixBlock[]> {
		private static final long serialVersionUID = -8819908424033945028L;

		@Override
		public MatrixBlock[] call(MatrixBlock v1) throws Exception {
			return ArrayUtils.toArray(v1);
		}
	}
	
	public static class MapJoinSignature2 implements Function<Tuple2<MatrixBlock,MatrixBlock>, MatrixBlock[]> {
		private static final long serialVersionUID = 7690448020081435520L;
		@Override
		public MatrixBlock[] call(Tuple2<MatrixBlock, MatrixBlock> v1) throws Exception {
			return ArrayUtils.toArray(v1._1(), v1._2());
		}
	}
	
	public static class MapJoinSignature3 implements Function<Tuple2<Tuple2<MatrixBlock,MatrixBlock>,MatrixBlock>, MatrixBlock[]> {
		private static final long serialVersionUID = -5222678882354280164L;
		@Override
		public MatrixBlock[] call(Tuple2<Tuple2<MatrixBlock, MatrixBlock>, MatrixBlock> v1) throws Exception {
			return ArrayUtils.toArray(v1._1()._1(), v1._1()._2(), v1._2());
		}
	}
}
