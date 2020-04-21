/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.tugraz.sysds.lops.WeightedCrossEntropy;
import org.tugraz.sysds.lops.WeightedCrossEntropy.WCeMMType;
import org.tugraz.sysds.lops.WeightedDivMM;
import org.tugraz.sysds.lops.WeightedDivMM.WDivMMType;
import org.tugraz.sysds.lops.WeightedDivMMR;
import org.tugraz.sysds.lops.WeightedSigmoid;
import org.tugraz.sysds.lops.WeightedSigmoid.WSigmoidType;
import org.tugraz.sysds.lops.WeightedSquaredLoss;
import org.tugraz.sysds.lops.WeightedSquaredLoss.WeightsType;
import org.tugraz.sysds.lops.WeightedSquaredLossR;
import org.tugraz.sysds.lops.WeightedUnaryMM;
import org.tugraz.sysds.lops.WeightedUnaryMM.WUMMType;
import org.tugraz.sysds.lops.WeightedUnaryMMR;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.instructions.cp.DoubleObject;
import org.tugraz.sysds.runtime.instructions.spark.data.LazyIterableIterator;
import org.tugraz.sysds.runtime.instructions.spark.data.PartitionedBroadcast;
import org.tugraz.sysds.runtime.instructions.spark.functions.FilterNonEmptyBlocksFunction;
import org.tugraz.sysds.runtime.instructions.spark.functions.ReplicateBlockFunction;
import org.tugraz.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.matrix.operators.QuaternaryOperator;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
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
		if(    WeightedSquaredLoss.OPCODE.equalsIgnoreCase(opcode)    //wsloss
			|| WeightedSquaredLossR.OPCODE.equalsIgnoreCase(opcode) )
		{
			boolean isRed = WeightedSquaredLossR.OPCODE.equalsIgnoreCase(opcode);
			
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
		else if(    WeightedUnaryMM.OPCODE.equalsIgnoreCase(opcode)    //wumm
			|| WeightedUnaryMMR.OPCODE.equalsIgnoreCase(opcode) )
		{
			boolean isRed = WeightedUnaryMMR.OPCODE.equalsIgnoreCase(opcode);
			
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
		else if(    WeightedDivMM.OPCODE.equalsIgnoreCase(opcode)    //wdivmm
				|| WeightedDivMMR.OPCODE.equalsIgnoreCase(opcode) )
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
			int addInput4 = (opcode.endsWith("wcemm")) ? 1 : 0;
			
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
		
			if( opcode.endsWith("wsigmoid") )
				return new QuaternarySPInstruction(new QuaternaryOperator(WSigmoidType.valueOf(parts[5])), in1, in2, in3, null, out, cacheU, cacheV, opcode, str);
			else if( opcode.endsWith("wcemm") ) {
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
		
		DataCharacteristics inMc = sec.getDataCharacteristics( input1.getName() );
		long rlen = inMc.getRows();
		long clen = inMc.getCols();
		int blen = inMc.getBlocksize();
		
		//pre-filter empty blocks (ultra-sparse matrices) for full aggregates
		//(map/redwsloss, map/redwcemm); safe because theses ops produce a scalar
		if( qop.wtype1 != null || qop.wtype4 != null ) {
			in = in.filter(new FilterNonEmptyBlocksFunction());
		}
		
		//map-side only operation (one rdd input, two broadcasts)
		if(    WeightedSquaredLoss.OPCODE.equalsIgnoreCase(getOpcode())
			|| WeightedSigmoid.OPCODE.equalsIgnoreCase(getOpcode())
			|| WeightedDivMM.OPCODE.equalsIgnoreCase(getOpcode()) 
			|| WeightedCrossEntropy.OPCODE.equalsIgnoreCase(getOpcode())
			|| WeightedUnaryMM.OPCODE.equalsIgnoreCase(getOpcode())) 
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

			//preparation of transposed and replicated U
			if( inU != null )
				inU = inU.flatMapToPair(new ReplicateBlockFunction(clen, blen, true));

			//preparation of transposed and replicated V
			if( inV != null )
				inV = inV.mapToPair(new TransposeFactorIndexesFunction())
				         .flatMapToPair(new ReplicateBlockFunction(rlen, blen, false));
			
			//functions calls w/ two rdd inputs		
			if( inU != null && inV == null && inW == null )
				out = in.join(inU)
				        .mapToPair(new RDDQuaternaryFunction2(qop, bc1, bc2));
			else if( inU == null && inV != null && inW == null )
				out = in.join(inV)
				        .mapToPair(new RDDQuaternaryFunction2(qop, bc1, bc2));
			else if( inU == null && inV == null && inW != null )
				out = in.join(inW)
				        .mapToPair(new RDDQuaternaryFunction2(qop, bc1, bc2));
			//function calls w/ three rdd inputs
			else if( inU != null && inV != null && inW == null )
				out = in.join(inU).join(inV)
				        .mapToPair(new RDDQuaternaryFunction3(qop, bc1, bc2));
			else if( inU != null && inV == null && inW != null )
				out = in.join(inU).join(inW)
				        .mapToPair(new RDDQuaternaryFunction3(qop, bc1, bc2));
			else if( inU == null && inV != null && inW != null )
				out = in.join(inV).join(inW)
				        .mapToPair(new RDDQuaternaryFunction3(qop, bc1, bc2));
			else if( inU == null && inV == null && inW == null ) {
				out = in.mapPartitionsToPair(new RDDQuaternaryFunction1(qop, bc1, bc2), false);
			}
			//function call w/ four rdd inputs
			else //need keys in case of wdivmm 
				out = in.join(inU).join(inV).join(inW)
				        .mapToPair(new RDDQuaternaryFunction4(qop));
			
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
			DoubleObject ret = new DoubleObject(tmp.getValue(0, 0));
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
				MatrixBlock blkOut = new MatrixBlock();
				MatrixBlock mbU = _pmU.getBlock((int)ixIn.getRowIndex(), 1);
				MatrixBlock mbV = _pmV.getBlock((int)ixIn.getColumnIndex(), 1);
				
				//execute core operation
				blkIn.quaternaryOperations(_qop, mbU, mbV, null, blkOut);
				
				//create return tuple
				MatrixIndexes ixOut = createOutputIndexes(ixIn);
				return new Tuple2<>(ixOut, blkOut);
			}
		}
	}

	private static class RDDQuaternaryFunction2 extends RDDQuaternaryBaseFunction //two rdd input
		implements PairFunction<Tuple2<MatrixIndexes, Tuple2<MatrixBlock,MatrixBlock>>, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = 7493974462943080693L;
		
		public RDDQuaternaryFunction2( QuaternaryOperator qop, PartitionedBroadcast<MatrixBlock> bcU, PartitionedBroadcast<MatrixBlock> bcV ) {
			super(qop, bcU, bcV);
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>> arg0) {
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn1 = arg0._2()._1();
			MatrixBlock blkIn2 = arg0._2()._2();
			MatrixBlock blkOut = new MatrixBlock();
			MatrixBlock mbU = (_pmU!=null)?_pmU.getBlock((int)ixIn.getRowIndex(), 1) : blkIn2;
			MatrixBlock mbV = (_pmV!=null)?_pmV.getBlock((int)ixIn.getColumnIndex(), 1) : blkIn2;
			MatrixBlock mbW = (_qop.hasFourInputs()) ? blkIn2 : null;
			
			//execute core operation
			blkIn1.quaternaryOperations(_qop, mbU, mbV, mbW, blkOut);
			
			//create return tuple
			MatrixIndexes ixOut = createOutputIndexes(ixIn);
			return new Tuple2<>(ixOut, blkOut);
		}
	}

	private static class RDDQuaternaryFunction3 extends RDDQuaternaryBaseFunction //three rdd input
		implements PairFunction<Tuple2<MatrixIndexes, Tuple2<Tuple2<MatrixBlock,MatrixBlock>,MatrixBlock>>, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = -2294086455843773095L;
		
		public RDDQuaternaryFunction3( QuaternaryOperator qop, PartitionedBroadcast<MatrixBlock> bcU, PartitionedBroadcast<MatrixBlock> bcV ) {
			super(qop, bcU, bcV);
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Tuple2<Tuple2<MatrixBlock, MatrixBlock>, MatrixBlock>> arg0) {
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn1 = arg0._2()._1()._1();
			MatrixBlock blkIn2 = arg0._2()._1()._2();
			MatrixBlock blkIn3 = arg0._2()._2();
			MatrixBlock blkOut = new MatrixBlock();
			MatrixBlock mbU = (_pmU!=null)?_pmU.getBlock((int)ixIn.getRowIndex(), 1) : blkIn2;
			MatrixBlock mbV = (_pmV!=null)?_pmV.getBlock((int)ixIn.getColumnIndex(), 1) : 
				              (_pmU!=null)? blkIn2 : blkIn3;
			MatrixBlock mbW = (_qop.hasFourInputs())? blkIn3 : null;
			
			//execute core operation
			blkIn1.quaternaryOperations(_qop, mbU, mbV, mbW, blkOut);
			
			//create return tuple
			MatrixIndexes ixOut = createOutputIndexes(ixIn);
			return new Tuple2<>(ixOut, blkOut);
		}
	}
	
	/**
	 * Note: never called for wsigmoid/wdivmm (only wsloss)
	 */
	private static class RDDQuaternaryFunction4 extends RDDQuaternaryBaseFunction //four rdd input
		implements PairFunction<Tuple2<MatrixIndexes,Tuple2<Tuple2<Tuple2<MatrixBlock,MatrixBlock>,MatrixBlock>,MatrixBlock>>,MatrixIndexes,MatrixBlock>
	{
		private static final long serialVersionUID = 7328911771600289250L;
		
		public RDDQuaternaryFunction4( QuaternaryOperator qop ) {
			super(qop, null, null);
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Tuple2<Tuple2<Tuple2<MatrixBlock, MatrixBlock>, MatrixBlock>, MatrixBlock>> arg0)
		{
			MatrixIndexes ixIn1 = arg0._1();
			MatrixBlock blkIn1 = arg0._2()._1()._1()._1();
			MatrixBlock mbU = arg0._2()._1()._1()._2();
			MatrixBlock mbV = arg0._2()._1()._2();
			MatrixBlock mbW = arg0._2()._2();
			MatrixBlock blkOut = new MatrixBlock();
			
			//execute core operation
			blkIn1.quaternaryOperations(_qop, mbU, mbV, mbW, blkOut);
			
			//create return tuple
			MatrixIndexes ixOut = createOutputIndexes(ixIn1);
			return new Tuple2<>(ixOut, blkOut);
		}
	}
	
	private static class TransposeFactorIndexesFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -2571724736131823708L;
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) {
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();

			//swap the matrix indexes
			MatrixIndexes ixOut = new MatrixIndexes(ixIn.getColumnIndex(), ixIn.getRowIndex());
			MatrixBlock blkOut = new MatrixBlock(blkIn);
			
			//output new tuple
			return new Tuple2<>(ixOut,blkOut);
		}
	}
}
