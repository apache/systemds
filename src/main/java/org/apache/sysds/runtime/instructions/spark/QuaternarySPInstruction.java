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
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.lops.WeightedCrossEntropy;
import org.apache.sysds.lops.WeightedCrossEntropy.WCeMMType;
import org.apache.sysds.lops.WeightedDivMM;
import org.apache.sysds.lops.WeightedDivMM.WDivMMType;
import org.apache.sysds.lops.WeightedDivMMR;
import org.apache.sysds.lops.WeightedSigmoid;
import org.apache.sysds.lops.WeightedSigmoid.WSigmoidType;
import org.apache.sysds.lops.WeightedSquaredLoss;
import org.apache.sysds.lops.WeightedSquaredLoss.WeightsType;
import org.apache.sysds.lops.WeightedSquaredLossR;
import org.apache.sysds.lops.WeightedUnaryMM;
import org.apache.sysds.lops.WeightedUnaryMM.WUMMType;
import org.apache.sysds.lops.WeightedUnaryMMR;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.spark.data.LazyIterableIterator;
import org.apache.sysds.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysds.runtime.instructions.spark.functions.FilterNonEmptyBlocksFunction;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.QuaternaryOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import scala.Tuple2;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;

public class QuaternarySPInstruction extends ComputationSPInstruction {
	private CPOperand _input4 = null;
	private boolean _cacheU = false;
	private boolean _cacheV = false;

	private QuaternarySPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4,
			CPOperand out, boolean cacheU, boolean cacheV, String opcode, String str) {
		super(SPType.Quaternary, op, in1, in2, in3, out, opcode, str);
		_input4 = in4;
		_cacheU = cacheU;
		_cacheV = cacheV;
	}

	public static QuaternarySPInstruction parseInstruction( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		//validity check
		if( !InstructionUtils.isDistQuaternaryOpcode(opcode) ) {
			throw new DMLRuntimeException("Quaternary.parseInstruction():: Unknown opcode " + opcode);
		}
		
		//instruction parsing
		if(    Opcodes.WEIGHTEDSQUAREDLOSS.toString().equalsIgnoreCase(opcode)    //wsloss
			|| Opcodes.WEIGHTEDSQUAREDLOSSR.toString().equalsIgnoreCase(opcode) )
		{
			boolean isRed = Opcodes.WEIGHTEDSQUAREDLOSSR.toString().equalsIgnoreCase(opcode);
			
			//check number of fields (4 inputs, output, type)
			if( isRed )
				InstructionUtils.checkNumFields ( parts, 8 );
			else
				InstructionUtils.checkNumFields ( parts, 6 );
			
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand in4 = new CPOperand(parts[4]);
			CPOperand out = new CPOperand(parts[5]);
			WeightsType wtype = WeightsType.valueOf(parts[6]);
			
			//in mappers always through distcache, in reducers through distcache/shuffle
			boolean cacheU = isRed ? Boolean.parseBoolean(parts[7]) : true;
			boolean cacheV = isRed ? Boolean.parseBoolean(parts[8]) : true;
			
			return new QuaternarySPInstruction(new QuaternaryOperator(wtype), in1, in2, in3, in4, out, cacheU, cacheV, opcode, str);	
		}
		else if(    Opcodes.WEIGHTEDUNARYMM.toString().equalsIgnoreCase(opcode)    //wumm
			|| Opcodes.WEIGHTEDUNARYMMR.toString().equalsIgnoreCase(opcode) )
		{
			boolean isRed = Opcodes.WEIGHTEDUNARYMMR.toString().equalsIgnoreCase(opcode);
			
			//check number of fields (4 inputs, output, type)
			if( isRed )
				InstructionUtils.checkNumFields ( parts, 8 );
			else
				InstructionUtils.checkNumFields ( parts, 6 );
			
			String uopcode = parts[1];
			CPOperand in1 = new CPOperand(parts[2]);
			CPOperand in2 = new CPOperand(parts[3]);
			CPOperand in3 = new CPOperand(parts[4]);
			CPOperand out = new CPOperand(parts[5]);
			WUMMType wtype = WUMMType.valueOf(parts[6]);
			
			//in mappers always through distcache, in reducers through distcache/shuffle
			boolean cacheU = isRed ? Boolean.parseBoolean(parts[7]) : true;
			boolean cacheV = isRed ? Boolean.parseBoolean(parts[8]) : true;
			
			return new QuaternarySPInstruction(new QuaternaryOperator(wtype, uopcode), in1, in2, in3, null, out, cacheU, cacheV, opcode, str);	
		}
		else if(    Opcodes.WEIGHTEDDIVMM.toString().equalsIgnoreCase(opcode)    //wdivmm
				|| Opcodes.WEIGHTEDDIVMMR.toString().equalsIgnoreCase(opcode) )
		{
			boolean isRed = opcode.startsWith("red");
			
			//check number of fields (4 inputs, output, type)
			if( isRed )
				InstructionUtils.checkNumFields( parts, 8 );
			else
				InstructionUtils.checkNumFields( parts, 6 );
			
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand in4 = new CPOperand(parts[4]);
			CPOperand out = new CPOperand(parts[5]);
			
			//in mappers always through distcache, in reducers through distcache/shuffle
			boolean cacheU = isRed ? Boolean.parseBoolean(parts[7]) : true;
			boolean cacheV = isRed ? Boolean.parseBoolean(parts[8]) : true;
		
			final WDivMMType wt = WDivMMType.valueOf(parts[6]);
			QuaternaryOperator qop = (wt.hasScalar() ? new QuaternaryOperator(wt, Double.parseDouble(in4.getName())) : new QuaternaryOperator(wt));
			return new QuaternarySPInstruction(qop, in1, in2, in3, in4, out, cacheU, cacheV, opcode, str);
		} 
		else //map/redwsigmoid, map/redwcemm
		{
			boolean isRed = opcode.startsWith("red");
			int addInput4 = (opcode.endsWith(Opcodes.WCEMM.toString())) ? 1 : 0;
			
			//check number of fields (3 or 4 inputs, output, type)
			if( isRed )
				InstructionUtils.checkNumFields( parts, 7 + addInput4 );
			else
				InstructionUtils.checkNumFields( parts, 5 + addInput4 );
			
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[4 + addInput4]);
			
			//in mappers always through distcache, in reducers through distcache/shuffle
			boolean cacheU = isRed ? Boolean.parseBoolean(parts[6 + addInput4]) : true;
			boolean cacheV = isRed ? Boolean.parseBoolean(parts[7 + addInput4]) : true;
		
			if( opcode.endsWith(Opcodes.WSIGMOID.toString()) )
				return new QuaternarySPInstruction(new QuaternaryOperator(WSigmoidType.valueOf(parts[5])), in1, in2, in3, null, out, cacheU, cacheV, opcode, str);
			else if( opcode.endsWith(Opcodes.WCEMM.toString()) ) {
				CPOperand in4 = new CPOperand(parts[4]);
				final WCeMMType wt = WCeMMType.valueOf(parts[6]);
				QuaternaryOperator qop = (wt.hasFourInputs() ? new QuaternaryOperator(wt, Double.parseDouble(in4.getName())) : new QuaternaryOperator(wt));
				return new QuaternarySPInstruction(qop, in1, in2, in3, in4, out, cacheU, cacheV, opcode, str);
			}
		}
		
		return null;
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		QuaternaryOperator qop = (QuaternaryOperator) _optr;
		
		//tracking of rdds and broadcasts (for lineage maintenance)
		ArrayList<String> rddVars = new ArrayList<>();
		ArrayList<String> bcVars = new ArrayList<>();

		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = null;
		
		//pre-filter empty blocks (ultra-sparse matrices) for full aggregates
		//(map/redwsloss, map/redwcemm); safe because theses ops produce a scalar
		if( qop.wtype1 != null || qop.wtype4 != null ) {
			in = in.filter(new FilterNonEmptyBlocksFunction());
		}
		
		//map-side only operation (one rdd input, two broadcasts)
		if(    Opcodes.WEIGHTEDSQUAREDLOSS.toString().equalsIgnoreCase(getOpcode())
			|| Opcodes.WEIGHTEDSIGMOID.toString().equalsIgnoreCase(getOpcode())
			|| Opcodes.WEIGHTEDDIVMM.toString().equalsIgnoreCase(getOpcode())
			|| Opcodes.WEIGHTEDCROSSENTROPY.toString().equalsIgnoreCase(getOpcode())
			|| Opcodes.WEIGHTEDUNARYMM.toString().equalsIgnoreCase(getOpcode()))
		{
			PartitionedBroadcast<MatrixBlock> bc1 = sec.getBroadcastForVariable( input2.getName() );
			PartitionedBroadcast<MatrixBlock> bc2 = sec.getBroadcastForVariable( input3.getName() );
			
			//partitioning-preserving mappartitions (key access required for broadcast loopkup)
			boolean noKeyChange = (qop.wtype3 == null || qop.wtype3.isBasic()); //only wdivmm changes keys
			out = in.mapPartitionsToPair(new RDDQuaternaryFunction1(qop, bc1, bc2), noKeyChange);
			
			rddVars.add( input1.getName() );
			bcVars.add( input2.getName() );
			bcVars.add( input3.getName() );
		}
		//reduce-side operation (two/three/four rdd inputs, zero/one/two broadcasts)
		else 
		{
			PartitionedBroadcast<MatrixBlock> bc1 = _cacheU ? sec.getBroadcastForVariable( input2.getName() ) : null;
			PartitionedBroadcast<MatrixBlock> bc2 = _cacheV ? sec.getBroadcastForVariable( input3.getName() ) : null;
			JavaPairRDD<MatrixIndexes,MatrixBlock> inU = (!_cacheU) ? sec.getBinaryMatrixBlockRDDHandleForVariable( input2.getName() ) : null;
			JavaPairRDD<MatrixIndexes,MatrixBlock> inV = (!_cacheV) ? sec.getBinaryMatrixBlockRDDHandleForVariable( input3.getName() ) : null;
			JavaPairRDD<MatrixIndexes,MatrixBlock> inW = (qop.hasFourInputs() && !_input4.isLiteral()) ? 
					sec.getBinaryMatrixBlockRDDHandleForVariable( _input4.getName() ) : null;

			//join X and W on original indexes if W existing
			JavaPairRDD<MatrixIndexes,MatrixBlock[]> tmp = (inW != null) ?
				in.join(inW).mapToPair(new ToArray()) :
				in.mapValues(mb -> new MatrixBlock[]{mb, null});
			
			//join lhs U on row-block indexes of X/W
			tmp = ( inU != null ) ?
				tmp.mapToPair(new ExtractIndexWith(true))
					.join(inU.mapToPair(new ExtractIndex(true))).mapToPair(new Unpack()) :
				tmp.mapValues(mb -> ArrayUtils.add(mb, null));
			
			//join rhs V on column-block indexes X/W (note V transposed input, so rows)
			tmp = ( inV != null ) ?
				tmp.mapToPair(new ExtractIndexWith(false))
					.join(inV.mapToPair(new ExtractIndex(true))).mapToPair(new Unpack()) :
				tmp.mapValues(mb -> ArrayUtils.add(mb, null));
			
			//execute quaternary block operations on joined inputs
			out = tmp.mapToPair(new RDDQuaternaryFunction2(qop, bc1, bc2));
			
			//keep variable names for lineage maintenance
			if( inU == null ) bcVars.add(input2.getName()); else rddVars.add(input2.getName());
			if( inV == null ) bcVars.add(input3.getName()); else rddVars.add(input3.getName());
			if( inW != null ) rddVars.add(_input4.getName());
		}
		
		//output handling, incl aggregation
		if( qop.wtype1 != null || qop.wtype4 != null ) //map/redwsloss, map/redwcemm
		{
			//full aggregate and cast to scalar
			MatrixBlock tmp = RDDAggregateUtils.sumStable(out);
			DoubleObject ret = new DoubleObject(tmp.get(0, 0));
			sec.setVariable(output.getName(), ret);
		}
		else //map/redwsigmoid, map/redwdivmm, map/redwumm 
		{
			//aggregation if required (map/redwdivmm)
			if( qop.wtype3 != null && !qop.wtype3.isBasic() )
				out = RDDAggregateUtils.sumByKeyStable(out, false);
				
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			//maintain lineage information for output rdd
			for( String rddVar : rddVars )
				sec.addLineageRDD(output.getName(), rddVar);
			for( String bcVar : bcVars )
				sec.addLineageBroadcast(output.getName(), bcVar);
			
			//update matrix characteristics
			updateOutputDataCharacteristics(sec, qop);
		}
	}

	private void updateOutputDataCharacteristics(SparkExecutionContext sec, QuaternaryOperator qop) {
		DataCharacteristics mcIn1 = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mcIn2 = sec.getDataCharacteristics(input2.getName());
		DataCharacteristics mcIn3 = sec.getDataCharacteristics(input3.getName());
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
		if( qop.wtype2 != null || qop.wtype5 != null ) {
			//output size determined by main input
			mcOut.set(mcIn1.getRows(), mcIn1.getCols(), mcIn1.getBlocksize(), mcIn1.getBlocksize());
		}
		else if(qop.wtype3 != null ) { //wdivmm
			long rank = qop.wtype3.isLeft() ? mcIn3.getCols() : mcIn2.getCols();
			DataCharacteristics mcTmp = qop.wtype3.computeOutputCharacteristics(mcIn1.getRows(), mcIn1.getCols(), rank);
			mcOut.set(mcTmp.getRows(), mcTmp.getCols(), mcIn1.getBlocksize(), mcIn1.getBlocksize());
		}
	}

	public CPOperand getInput4() {
		return _input4;
	}

	private abstract static class RDDQuaternaryBaseFunction implements Serializable
	{
		private static final long serialVersionUID = -3175397651350954930L;
		
		protected QuaternaryOperator _qop = null;
		protected PartitionedBroadcast<MatrixBlock> _pmU = null;
		protected PartitionedBroadcast<MatrixBlock> _pmV = null;
		
		public RDDQuaternaryBaseFunction( QuaternaryOperator qop, PartitionedBroadcast<MatrixBlock> bcU, PartitionedBroadcast<MatrixBlock> bcV ) {
			_qop = qop;
			_pmU = bcU;
			_pmV = bcV;
		}

		protected MatrixIndexes createOutputIndexes(MatrixIndexes in) {
			if( _qop.wtype3 != null && !_qop.wtype3.isBasic() ){ //key change
				boolean left = _qop.wtype3.isLeft();
				return new MatrixIndexes(left?in.getColumnIndex():in.getRowIndex(), 1);
			}
			return in;
		}
	}

	private static class RDDQuaternaryFunction1 extends RDDQuaternaryBaseFunction //one rdd input
		implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes, MatrixBlock>>, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = -8209188316939435099L;
		
		public RDDQuaternaryFunction1( QuaternaryOperator qop, PartitionedBroadcast<MatrixBlock> bcU, PartitionedBroadcast<MatrixBlock> bcV ) {
			super(qop, bcU, bcV);
		}
	
		@Override
		public LazyIterableIterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg) {
			return new RDDQuaternaryPartitionIterator(arg);
		}
		
		private class RDDQuaternaryPartitionIterator extends LazyIterableIterator<Tuple2<MatrixIndexes, MatrixBlock>>
		{
			public RDDQuaternaryPartitionIterator(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> in) {
				super(in);
			}
			
			@Override
			protected Tuple2<MatrixIndexes, MatrixBlock> computeNext(Tuple2<MatrixIndexes, MatrixBlock> arg) {
				MatrixIndexes ixIn = arg._1();
				MatrixBlock blkIn = arg._2();
				MatrixBlock mbU = _pmU.getBlock((int)ixIn.getRowIndex(), 1);
				MatrixBlock mbV = _pmV.getBlock((int)ixIn.getColumnIndex(), 1);
				
				//execute core operation
				MatrixBlock blkOut = blkIn.quaternaryOperations(_qop, mbU, mbV, null, new MatrixBlock());
				
				//create return tuple
				MatrixIndexes ixOut = createOutputIndexes(ixIn);
				return new Tuple2<>(ixOut, blkOut);
			}
		}
	}

	private static class RDDQuaternaryFunction2 extends RDDQuaternaryBaseFunction //two rdd input
		implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock[]>, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = 7493974462943080693L;
		
		public RDDQuaternaryFunction2( QuaternaryOperator qop, PartitionedBroadcast<MatrixBlock> bcU, PartitionedBroadcast<MatrixBlock> bcV ) {
			super(qop, bcU, bcV);
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock[]> arg0) {
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock[] blks = arg0._2();
			MatrixBlock mbU = (_pmU!=null)?_pmU.getBlock((int)ixIn.getRowIndex(), 1) : blks[2];
			MatrixBlock mbV = (_pmV!=null)?_pmV.getBlock((int)ixIn.getColumnIndex(), 1) : blks[3];
			MatrixBlock mbW = (_qop.hasFourInputs()) ? blks[1] : null;
			
			//execute core operation
			MatrixBlock blkOut = blks[0].quaternaryOperations(_qop, mbU, mbV, mbW, new MatrixBlock());
			
			//create return tuple
			MatrixIndexes ixOut = createOutputIndexes(ixIn);
			return new Tuple2<>(ixOut, blkOut);
		}
	}

	private static class ExtractIndex implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, Long, MatrixBlock> {
		private static final long serialVersionUID = -6542246824481788376L;
		private final boolean _row;
		public ExtractIndex(boolean row) {
			_row = row;
		}
		@Override
		public Tuple2<Long, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg) throws Exception {
			return new Tuple2<>(_row?arg._1().getRowIndex():arg._1().getColumnIndex(), arg._2());
		}
	}
	
	private static class ExtractIndexWith implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock[]>, Long, Tuple2<MatrixIndexes,MatrixBlock[]>> {
		private static final long serialVersionUID = -966212318512764461L;
		private final boolean _row;
		public ExtractIndexWith(boolean row) {
			_row = row;
		}
		@Override
		public Tuple2<Long, Tuple2<MatrixIndexes, MatrixBlock[]>> call(Tuple2<MatrixIndexes, MatrixBlock[]> arg)
			throws Exception
		{
			return new Tuple2<>(_row?arg._1().getRowIndex():arg._1().getColumnIndex(), arg);
		}
	}
	
	private static class ToArray implements PairFunction<Tuple2<MatrixIndexes,Tuple2<MatrixBlock,MatrixBlock>>, MatrixIndexes, MatrixBlock[]> {
		private static final long serialVersionUID = -4856316007590144978L;

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock[]> call(Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>> arg)
			throws Exception 
		{
			return new Tuple2<>(arg._1(), new MatrixBlock[]{arg._2()._1(),arg._2()._2()});
		}
	}
	
	private static class Unpack implements PairFunction<Tuple2<Long, Tuple2<Tuple2<MatrixIndexes,MatrixBlock[]>,MatrixBlock>>, MatrixIndexes, MatrixBlock[]> {
		private static final long serialVersionUID = 3812660351709830714L;
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock[]> call(
			Tuple2<Long, Tuple2<Tuple2<MatrixIndexes, MatrixBlock[]>, MatrixBlock>> arg) throws Exception
		{
			return new Tuple2<>(arg._2()._1()._1(),                    //matrix indexes
				ArrayUtils.addAll(arg._2()._1()._2(), arg._2()._2())); //array of matrix blocks
		}
	}
}
