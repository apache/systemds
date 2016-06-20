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

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;

import org.apache.spark.Accumulator;
import org.apache.spark.AccumulatorParam;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.FrameObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.ParameterizedBuiltinSPInstruction.RDDTransformApplyFunction;
import org.apache.sysml.runtime.instructions.spark.ParameterizedBuiltinSPInstruction.RDDTransformApplyOffsetFunction;
import org.apache.sysml.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.apache.sysml.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysml.runtime.io.FrameReader;
import org.apache.sysml.runtime.io.FrameReaderFactory;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.transform.RecodeAgent;
import org.apache.sysml.runtime.transform.encode.Encoder;
import org.apache.sysml.runtime.transform.encode.EncoderComposite;
import org.apache.sysml.runtime.transform.encode.EncoderFactory;
import org.apache.sysml.runtime.transform.meta.TfMetaUtils;
import org.apache.sysml.runtime.transform.meta.TfOffsetMap;

import scala.Tuple2;


public class MultiReturnParameterizedBuiltinSPInstruction extends ComputationSPInstruction 
{
	protected ArrayList<CPOperand> _outputs;
	
	public MultiReturnParameterizedBuiltinSPInstruction(Operator op, CPOperand input1, CPOperand input2, ArrayList<CPOperand> outputs, String opcode, String istr ) {
		super(op, input1, input2, outputs.get(0), opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.MultiReturnBuiltin;
		_outputs = outputs;
	}
	
	public CPOperand getOutput(int i) {
		return _outputs.get(i);
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MultiReturnParameterizedBuiltinSPInstruction parseInstruction( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		ArrayList<CPOperand> outputs = new ArrayList<CPOperand>();
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase("transformencode") ) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			outputs.add ( new CPOperand(parts[3], ValueType.DOUBLE, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[4], ValueType.STRING, DataType.FRAME) );
			return new MultiReturnParameterizedBuiltinSPInstruction(null, in1, in2, outputs, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + opcode);
		}

	}

	@Override 
	@SuppressWarnings("unchecked")
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException 
	{
		SparkExecutionContext sec = (SparkExecutionContext) ec;
		
		try
		{
			//get input RDD and meta data
			FrameObject fo = sec.getFrameObject(input1.getName());
			FrameObject fometa = sec.getFrameObject(_outputs.get(1).getName());
			JavaPairRDD<Long,FrameBlock> in = (JavaPairRDD<Long,FrameBlock>)
					sec.getRDDHandleForFrameObject(fo, InputInfo.BinaryBlockInputInfo);
			String spec = ec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral()).getStringValue();
			MatrixCharacteristics mcIn = sec.getMatrixCharacteristics(input1.getName());
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			
			//step 1: build transform meta data
			Encoder encoderBuild = EncoderFactory.createEncoder(spec, 
					fo.getSchema(), (int)fo.getNumColumns(), null);
			
			Accumulator<Long> accMax = sec.getSparkContext().accumulator(0L, new MaxAcc()); 
			in.mapPartitionsToPair(new TransformEncodeBuildFunction(encoderBuild))
			  .distinct().groupByKey()
			  .flatMap(new TransformEncodeGroupFunction(accMax))
			  .saveAsTextFile(fometa.getFileName()); //trigger eval
			
			//reuse multi-threaded reader 
			FrameReader reader = FrameReaderFactory.createFrameReader(InputInfo.TextCellInputInfo);
			FrameBlock meta = reader.readFrameFromHDFS(fometa.getFileName(), accMax.value(), fo.getNumColumns());
			meta.recomputeColumnCardinality(); //recompute num distinct items per column
			
			//step 2: transform apply (similar to spark transformapply)
			//compute omit offset map for block shifts
			TfOffsetMap omap = null;
			if( TfMetaUtils.containsOmitSpec(spec) ) {
				omap = new TfOffsetMap(SparkUtils.toIndexedLong(in.mapToPair(
					new RDDTransformApplyOffsetFunction(spec)).collect()));
			}
				
			//create encoder broadcast (avoiding replication per task) 
			Encoder encoder = EncoderFactory.createEncoder(spec, 
					fo.getSchema(), (int)fo.getNumColumns(), meta);
			mcOut.setDimension(mcIn.getRows()-((omap!=null)?omap.getNumRmRows():0), encoder.getNumCols()); 
			Broadcast<Encoder> bmeta = sec.getSparkContext().broadcast(encoder);
			Broadcast<TfOffsetMap> bomap = (omap!=null) ? sec.getSparkContext().broadcast(omap) : null;
			
			//execute transform apply
			JavaPairRDD<Long,FrameBlock> tmp = in
					.mapToPair(new RDDTransformApplyFunction(bmeta, bomap));
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = FrameRDDConverterUtils
					.binaryBlockToMatrixBlock(tmp, mcOut, mcOut);
			
			//set output and maintain lineage/output characteristics
			sec.setRDDHandleForVariable(_outputs.get(0).getName(), out);
			sec.addLineageRDD(_outputs.get(0).getName(), input1.getName());
			sec.setFrameOutput(_outputs.get(1).getName(), meta);
		}
		catch(IOException ex) {
			throw new RuntimeException(ex);
		}
	}
	
	/**
	 * 
	 */
	public static class TransformEncodeBuildFunction implements PairFlatMapFunction<Iterator<Tuple2<Long, FrameBlock>>, Integer, String>
	{
		private static final long serialVersionUID = 6336375833412029279L;

		private Encoder _encoder = null;
		
		public TransformEncodeBuildFunction(Encoder encoder) {
			_encoder = encoder;
		}
		
		@Override
		public Iterable<Tuple2<Integer, String>> call(Iterator<Tuple2<Long, FrameBlock>> iter)
			throws Exception 
		{
			//build meta data (e.g., recode maps)
			while( iter.hasNext() ) {
				_encoder.build(iter.next()._2());	
			}
			
			//output recode maps as columnID - token pairs
			ArrayList<Tuple2<Integer,String>> ret = new ArrayList<Tuple2<Integer,String>>();
			if( _encoder instanceof EncoderComposite )
				for( Encoder cEncoder : ((EncoderComposite)_encoder).getEncoders() )
					if( cEncoder instanceof RecodeAgent ) {
						RecodeAgent ra = (RecodeAgent) cEncoder;
						HashMap<Integer,HashMap<String,Long>> tmp = ra.getCPRecodeMaps();
						for( Entry<Integer,HashMap<String,Long>> e1 : tmp.entrySet() )
							for( String token : e1.getValue().keySet() )
								ret.add(new Tuple2<Integer,String>(e1.getKey(), token));
					}
				
			return ret;
		}
	}
	
	/**
	 * 
	 */
	public static class TransformEncodeGroupFunction implements FlatMapFunction<Tuple2<Integer, Iterable<String>>, String>
	{
		private static final long serialVersionUID = -1034187226023517119L;

		private Accumulator<Long> _accMax = null;
		
		public TransformEncodeGroupFunction( Accumulator<Long> accMax ) {
			_accMax = accMax;
		}
		
		@Override
		public Iterable<String> call(Tuple2<Integer, Iterable<String>> arg0)
			throws Exception 
		{
			String colID = String.valueOf(arg0._1());
			Iterator<String> iter = arg0._2().iterator();
			
			ArrayList<String> ret = new ArrayList<String>();
			StringBuilder sb = new StringBuilder();
			long rowID = 1;
			while( iter.hasNext() ) {
				sb.append(rowID);
				sb.append(' ');
				sb.append(colID);
				sb.append(' ');
				sb.append(RecodeAgent.constructRecodeMapEntry(iter.next(), rowID));
				ret.add(sb.toString());
				sb.setLength(0); 
				rowID++;
			}
			_accMax.add(rowID-1);
			
			return ret;
		}
	}
	
	/**
	 * 
	 */
	private static class MaxAcc implements AccumulatorParam<Long>, Serializable 
	{
		private static final long serialVersionUID = -3739727823287550826L;

		@Override
		public Long addInPlace(Long arg0, Long arg1) {
			return Math.max(arg0, arg1);
		}

		@Override
		public Long zero(Long arg0) {
			return arg0;
		}

		@Override
		public Long addAccumulator(Long arg0, Long arg1) {
			return Math.max(arg0, arg1);	
		}
	}
}
