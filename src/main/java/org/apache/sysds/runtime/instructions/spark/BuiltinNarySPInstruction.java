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
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.spark.AppendGSPInstruction.ShiftMatrix;
import org.apache.sysds.runtime.instructions.spark.functions.MapInputSignature;
import org.apache.sysds.runtime.instructions.spark.functions.MapJoinSignature;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.lineage.LineageTraceable;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.SimpleOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.UtilFunctions;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static org.apache.sysds.hops.BinaryOp.AppendMethod.MR_MAPPEND;
import static org.apache.sysds.hops.BinaryOp.AppendMethod.MR_RAPPEND;
import static org.apache.sysds.hops.OptimizerUtils.DEFAULT_FRAME_BLOCKSIZE;
import static org.apache.sysds.runtime.instructions.spark.FrameAppendMSPInstruction.appendFrameMSP;
import static org.apache.sysds.runtime.instructions.spark.FrameAppendRSPInstruction.appendFrameRSP;

public class BuiltinNarySPInstruction extends SPInstruction implements LineageTraceable
{
	public CPOperand[] inputs;
	public CPOperand output;
	
	protected BuiltinNarySPInstruction(CPOperand[] in, CPOperand out, String opcode, String istr) {
		super(SPType.BuiltinNary, opcode, istr);
		inputs = in;
		output = out;
	}

	public static BuiltinNarySPInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand output = new CPOperand(parts[parts.length - 1]);
		CPOperand[] inputs = null;
		inputs = new CPOperand[parts.length - 2];
		for (int i = 1; i < parts.length-1; i++)
			inputs[i-1] = new CPOperand(parts[i]);
		return new BuiltinNarySPInstruction(inputs, output, opcode, str);
	}

	@Override 
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		DataCharacteristics dcout = null;
		boolean inputIsMatrix = inputs[0].isMatrix();

		
		if( getOpcode().equals(Opcodes.CBIND.toString()) || getOpcode().equals(Opcodes.RBIND.toString()) ) {
			//compute output characteristics
			boolean cbind = getOpcode().equals(Opcodes.CBIND.toString());
			dcout = computeAppendOutputDataCharacteristics(sec, inputs, cbind);
			if(inputIsMatrix){
				//get consolidated input via union over shifted and padded inputs
				DataCharacteristics off = new MatrixCharacteristics(0, 0, dcout.getBlocksize(), 0);
				for( CPOperand input : inputs ) {
					DataCharacteristics mcIn = sec.getDataCharacteristics(input.getName());
					JavaPairRDD<MatrixIndexes, MatrixBlock> in = sec
							.getBinaryMatrixBlockRDDHandleForVariable(input.getName())
							.flatMapToPair(new ShiftMatrix(off, mcIn, cbind))
							.mapToPair(new PadBlocksFunction(dcout)); //just padding
					out = (out != null) ? out.union(in) : in;
					updateAppendDataCharacteristics(mcIn, off, cbind);
				}
				//aggregate partially overlapping blocks w/ single shuffle
				int numPartOut = SparkUtils.getNumPreferredPartitions(dcout);
				out = RDDAggregateUtils.mergeByKey(out, numPartOut, false);
			}
			//FRAME
			else {
				JavaPairRDD<Long,FrameBlock> outFrame = 
					sec.getFrameBinaryBlockRDDHandleForVariable( inputs[0].getName() );
				dcout = new MatrixCharacteristics(sec.getDataCharacteristics(inputs[0].getName()));
				FrameObject fo = new FrameObject(sec.getFrameObject(inputs[0].getName()));
				boolean[] broadcasted = new boolean[inputs.length];
				broadcasted[0] = false;

				for(int i = 1; i < inputs.length; i++){
					DataCharacteristics dcIn = sec.getDataCharacteristics(inputs[i].getName());
					final int blk_size = dcout.getBlocksize() <= 0 ? DEFAULT_FRAME_BLOCKSIZE : dcout.getBlocksize();

					broadcasted[i] = BinaryOp.FORCED_APPEND_METHOD == MR_MAPPEND
						|| BinaryOp.FORCED_APPEND_METHOD == null && cbind && dcIn.getCols() <= blk_size 
							&& OptimizerUtils.checkSparkBroadcastMemoryBudget(
								dcout.getCols(), dcIn.getCols(), blk_size, dcIn.getNonZeros());

					//easy case: broadcast & map
					if(broadcasted[i]){
						outFrame = appendFrameMSP(outFrame, sec.getBroadcastForFrameVariable(inputs[i].getName()));
					}
					//general case for frames:
					else{
						if(BinaryOp.FORCED_APPEND_METHOD != null && BinaryOp.FORCED_APPEND_METHOD != MR_RAPPEND)
							throw new DMLRuntimeException("Forced append type ["
								+BinaryOp.FORCED_APPEND_METHOD+"] is not supported for frames");

						JavaPairRDD<Long,FrameBlock> in2 = 
							sec.getFrameBinaryBlockRDDHandleForVariable(inputs[i].getName() );
						outFrame = appendFrameRSP(outFrame, in2, dcout.getRows(), cbind);
					}
					updateAppendDataCharacteristics(dcIn, dcout, cbind);
					if(cbind)
						fo.setSchema(fo.mergeSchemas(sec.getFrameObject(inputs[i].getName())));
				}

				//set output RDD and add lineage
				sec.getDataCharacteristics(output.getName()).set(dcout);
				sec.setRDDHandleForVariable(output.getName(), outFrame);
				sec.getFrameObject(output.getName()).setSchema(fo.getSchema());
				for( int i = 0; i < inputs.length; i++)
					if(broadcasted[i])
						sec.addLineageBroadcast(output.getName(), inputs[i].getName());
					else
						sec.addLineageRDD(output.getName(), inputs[i].getName());
				return;
			}
		}
		else if( ArrayUtils.contains(new String[]{Opcodes.NMIN.toString(),Opcodes.NMAX.toString(),Opcodes.NP.toString(),Opcodes.NM.toString()}, getOpcode()) ) {
			//compute output characteristics
			dcout = computeMinMaxOutputDataCharacteristics(sec, inputs);
			
			//get scalars and consolidated input via join
			List<ScalarObject> scalars = sec.getScalarInputs(inputs);
			JavaPairRDD<MatrixIndexes, MatrixBlock[]> in = null;
			for( CPOperand input : inputs ) {
				if( !input.getDataType().isMatrix() ) continue;
				JavaPairRDD<MatrixIndexes, MatrixBlock> tmp = sec
					.getBinaryMatrixBlockRDDHandleForVariable(input.getName());
				in = (in == null) ? tmp.mapValues(new MapInputSignature()) :
					in.join(tmp).mapValues(new MapJoinSignature());
			}
			
			//compute nary min/max (partitioning-preserving)
			out = in.mapValues(new MinMaxAddMultFunction(getOpcode(), scalars));
		}
		
		//set output RDD and add lineage
		sec.getDataCharacteristics(output.getName()).set(dcout);
		sec.setRDDHandleForVariable(output.getName(), out);
		for( CPOperand input : inputs )
			if( !input.isScalar() )
				sec.addLineageRDD(output.getName(), input.getName());
	}

	@SuppressWarnings("unused")
	private static class AlignBlkTask implements PairFlatMapFunction<Tuple2<Long, FrameBlock>, Long, FrameBlock> {
		private static final long serialVersionUID = 1333460067852261573L;
		long max_rows;

		public AlignBlkTask(long rows) {
			max_rows = rows;
		}

		@Override
		public Iterator<Tuple2<Long, FrameBlock>> call(Tuple2<Long, FrameBlock> longFrameBlockTuple2) throws Exception {
			Long index = longFrameBlockTuple2._1;
			FrameBlock fb = longFrameBlockTuple2._2;
			ArrayList<Tuple2<Long, FrameBlock>> list = new ArrayList<Tuple2<Long, FrameBlock>>();
			//single output block
			if(max_rows <= DEFAULT_FRAME_BLOCKSIZE){
				FrameBlock fbout = new FrameBlock(fb.getSchema());
				fbout.ensureAllocatedColumns((int) max_rows);
				fbout = fbout.leftIndexingOperations(fb,index.intValue() - 1, index.intValue() + fb.getNumRows() - 2,0, fb.getNumColumns()-1, null );
				list.add(new Tuple2<>(1L, fbout));
			} else {
				throw new NotImplementedException("Other Alignment strategies need to be implemented");
				//long aligned_index = (index / DEFAULT_FRAME_BLOCKSIZE)*OptimizerUtils.DEFAULT_FRAME_BLOCKSIZE+1;
				//list.add(new Tuple2<>(index / DEFAULT_FRAME_BLOCKSIZE + 1, fb));
			}

			return list.iterator();
		}
	}

	private static DataCharacteristics computeAppendOutputDataCharacteristics(SparkExecutionContext sec, CPOperand[] inputs, boolean cbind) {
		DataCharacteristics mcIn1 = sec.getDataCharacteristics(inputs[0].getName());
		DataCharacteristics mcOut = new MatrixCharacteristics(0, 0, mcIn1.getBlocksize(), 0);
		for( CPOperand input : inputs ) {
			DataCharacteristics mcIn = sec.getDataCharacteristics(input.getName());
			updateAppendDataCharacteristics(mcIn, mcOut, cbind);
		}
		return mcOut;
	}
	
	private static void updateAppendDataCharacteristics(DataCharacteristics in, DataCharacteristics out, boolean cbind) {
		out.setDimension(cbind ? Math.max(out.getRows(), in.getRows()) : out.getRows()+in.getRows(),
			cbind ? out.getCols()+in.getCols() : Math.max(out.getCols(), in.getCols()));
		out.setNonZeros((out.getNonZeros()!=-1 && in.dimsKnown(true)) ? out.getNonZeros()+in.getNonZeros() : -1);
	}
	
	private static DataCharacteristics computeMinMaxOutputDataCharacteristics(SparkExecutionContext sec, CPOperand[] inputs) {
		DataCharacteristics mcOut = new MatrixCharacteristics();
		for( CPOperand input : inputs ) {
			if( !input.getDataType().isMatrix() ) continue;
			DataCharacteristics mcIn = sec.getDataCharacteristics(input.getName());
			mcOut.setRows(Math.max(mcOut.getRows(), mcIn.getRows()));
			mcOut.setCols(Math.max(mcOut.getCols(), mcIn.getCols()));
			mcOut.setBlocksize(mcIn.getBlocksize());
		}
		return mcOut;
	}
	
	public static class PadBlocksFunction implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = 1291358959908299855L;
		
		private final DataCharacteristics _mcOut;
		
		public PadBlocksFunction(DataCharacteristics mcOut) {
			_mcOut = mcOut;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) throws Exception {
			MatrixIndexes ix = arg0._1();
			MatrixBlock mb = arg0._2();
			int brlen = UtilFunctions.computeBlockSize(_mcOut.getRows(), ix.getRowIndex(), _mcOut.getBlocksize());
			int bclen = UtilFunctions.computeBlockSize(_mcOut.getCols(), ix.getColumnIndex(), _mcOut.getBlocksize());
			
			//check for pass-through
			if( brlen == mb.getNumRows() && bclen == mb.getNumColumns() )
				return arg0;
			
			//cbind or rbind to pad to right blocksize
			if( brlen > mb.getNumRows() ) //rbind
				mb = mb.append(new MatrixBlock(brlen-mb.getNumRows(),bclen,true), new MatrixBlock(), false);
			else if( bclen > mb.getNumColumns() ) //cbind
				mb = mb.append(new MatrixBlock(brlen,bclen-mb.getNumColumns(),true), new MatrixBlock(), true);
			return new Tuple2<>(ix, mb);
		}
	}
	
	private static class MinMaxAddMultFunction implements Function<MatrixBlock[], MatrixBlock> {
		private static final long serialVersionUID = -4227447915387484397L;
		
		private final SimpleOperator _op;
		private final ScalarObject[] _scalars;

		public MinMaxAddMultFunction(String opcode, List<ScalarObject> scalars) {
			_scalars = scalars.toArray(new ScalarObject[0]);
			_op = new SimpleOperator(opcode.equals(Opcodes.NP.toString()) ? Plus.getPlusFnObject() :
					opcode.equals(Opcodes.NM.toString()) ? Multiply.getMultiplyFnObject() :
							Builtin.getBuiltinFnObject(opcode.substring(1)));
		}
		
		@Override
		public MatrixBlock call(MatrixBlock[] v1) throws Exception {
			return MatrixBlock.naryOperations(_op, v1, _scalars, new MatrixBlock());
		}
	}
	
	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		return Pair.of(output.getName(), new LineageItem(getOpcode(),
			LineageItemUtils.getLineage(ec, inputs)));
	}
}
