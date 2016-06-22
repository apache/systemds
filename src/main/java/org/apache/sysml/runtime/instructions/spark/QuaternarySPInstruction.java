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


import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import org.apache.sysml.lops.WeightedCrossEntropy;
import org.apache.sysml.lops.WeightedDivMM;
import org.apache.sysml.lops.WeightedDivMM.WDivMMType;
import org.apache.sysml.lops.WeightedDivMMR;
import org.apache.sysml.lops.WeightedSigmoid;
import org.apache.sysml.lops.WeightedSquaredLoss;
import org.apache.sysml.lops.WeightedSquaredLossR;
import org.apache.sysml.lops.WeightedSigmoid.WSigmoidType;
import org.apache.sysml.lops.WeightedSquaredLoss.WeightsType;
import org.apache.sysml.lops.WeightedCrossEntropy.WCeMMType;
import org.apache.sysml.lops.WeightedUnaryMM;
import org.apache.sysml.lops.WeightedUnaryMM.WUMMType;
import org.apache.sysml.lops.WeightedUnaryMMR;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.DoubleObject;
import org.apache.sysml.runtime.instructions.spark.data.LazyIterableIterator;
import org.apache.sysml.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysml.runtime.instructions.spark.functions.FilterNonEmptyBlocksFunction;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.QuaternaryOperator;

/**
 * 
 */
public class QuaternarySPInstruction extends ComputationSPInstruction 
{
	
	private CPOperand _input4 = null;
	private boolean _cacheU = false;
	private boolean _cacheV = false;
	
	public QuaternarySPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand out, boolean cacheU, boolean cacheV, String opcode, String str)
	{
		super(op, in1, in2, in3, out, opcode, str);
		_sptype = SPINSTRUCTION_TYPE.Quaternary;
		_input4 = in4;
		
		_cacheU = cacheU;
		_cacheV = cacheV;
	}

	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static QuaternarySPInstruction parseInstruction( String str ) 
		throws DMLRuntimeException 
	{
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
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException
	{	
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		QuaternaryOperator qop = (QuaternaryOperator) _optr;
		
		//tracking of rdds and broadcasts (for lineage maintenance)
		ArrayList<String> rddVars = new ArrayList<String>();
		ArrayList<String> bcVars = new ArrayList<String>();

		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = null;
		
		MatrixCharacteristics inMc = sec.getMatrixCharacteristics( input1.getName() );
		long rlen = inMc.getRows();
		long clen = inMc.getCols();
		int brlen = inMc.getRowsPerBlock();
		int bclen = inMc.getColsPerBlock();
		
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
			JavaPairRDD<MatrixIndexes,MatrixBlock> inU = (!_cacheU) ? sec.getBinaryBlockRDDHandleForVariable( input2.getName() ) : null;
			JavaPairRDD<MatrixIndexes,MatrixBlock> inV = (!_cacheV) ? sec.getBinaryBlockRDDHandleForVariable( input3.getName() ) : null;
			JavaPairRDD<MatrixIndexes,MatrixBlock> inW = (qop.hasFourInputs() && !_input4.isLiteral()) ? 
					sec.getBinaryBlockRDDHandleForVariable( _input4.getName() ) : null;

			//preparation of transposed and replicated U
			if( inU != null )
				inU = inU.flatMapToPair(new ReplicateBlocksFunction(clen, bclen, true));

			//preparation of transposed and replicated V
			if( inV != null )
				inV = inV.mapToPair(new TransposeFactorIndexesFunction())
				         .flatMapToPair(new ReplicateBlocksFunction(rlen, brlen, false));
			
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
				out = RDDAggregateUtils.sumByKeyStable( out );
				
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			//maintain lineage information for output rdd
			for( String rddVar : rddVars )
				sec.addLineageRDD(output.getName(), rddVar);
			for( String bcVar : bcVars )
				sec.addLineageBroadcast(output.getName(), bcVar);
			
			//update matrix characteristics
			updateOutputMatrixCharacteristics(sec, qop);
		}
	}
	
	/**
	 * 
	 * @param sec
	 * @param qop
	 * @throws DMLRuntimeException 
	 */
	private void updateOutputMatrixCharacteristics(SparkExecutionContext sec, QuaternaryOperator qop) 
		throws DMLRuntimeException
	{
		MatrixCharacteristics mcIn1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mcIn2 = sec.getMatrixCharacteristics(input2.getName());
		MatrixCharacteristics mcIn3 = sec.getMatrixCharacteristics(input3.getName());
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		
		if( qop.wtype2 != null || qop.wtype5 != null ) {
			//output size determined by main input
			mcOut.set(mcIn1.getRows(), mcIn1.getCols(), mcIn1.getRowsPerBlock(), mcIn1.getColsPerBlock());
		}
		else if(qop.wtype3 != null ) { //wdivmm
			long rank = qop.wtype3.isLeft() ? mcIn3.getCols() : mcIn2.getCols();
			MatrixCharacteristics mcTmp = qop.wtype3.computeOutputCharacteristics(mcIn1.getRows(), mcIn1.getCols(), rank);		
			mcOut.set(mcTmp.getRows(), mcTmp.getCols(), mcIn1.getRowsPerBlock(), mcIn1.getColsPerBlock());		
		}
	}
	
	/**
	 * 
	 */
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

		protected MatrixIndexes createOutputIndexes(MatrixIndexes in) 
		{
			if( _qop.wtype3 != null && !_qop.wtype3.isBasic() ){ //key change
				boolean left = _qop.wtype3.isLeft();
				return new MatrixIndexes(left?in.getColumnIndex():in.getRowIndex(), 1);
			}				
			return in;
		}
	}
	
	/**
	 * 
	 */
	private static class RDDQuaternaryFunction1 extends RDDQuaternaryBaseFunction //one rdd input
		implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes, MatrixBlock>>, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = -8209188316939435099L;
		
		public RDDQuaternaryFunction1( QuaternaryOperator qop, PartitionedBroadcast<MatrixBlock> bcU, PartitionedBroadcast<MatrixBlock> bcV ) 
			throws DMLRuntimeException
		{
			super(qop, bcU, bcV);
		}
	
		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg)
			throws Exception 
		{
			return new RDDQuaternaryPartitionIterator(arg);
		}
		
		private class RDDQuaternaryPartitionIterator extends LazyIterableIterator<Tuple2<MatrixIndexes, MatrixBlock>>
		{
			public RDDQuaternaryPartitionIterator(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> in) {
				super(in);
			}
			
			@Override
			protected Tuple2<MatrixIndexes, MatrixBlock> computeNext(Tuple2<MatrixIndexes, MatrixBlock> arg) 
				throws Exception 
			{
				MatrixIndexes ixIn = arg._1();
				MatrixBlock blkIn = arg._2();
				MatrixBlock blkOut = new MatrixBlock();
				
				MatrixBlock mbU = _pmU.getBlock((int)ixIn.getRowIndex(), 1);
				MatrixBlock mbV = _pmV.getBlock((int)ixIn.getColumnIndex(), 1);
				
				//execute core operation
				blkIn.quaternaryOperations(_qop, mbU, mbV, null, blkOut);
				
				//create return tuple
				MatrixIndexes ixOut = createOutputIndexes(ixIn);
				return new Tuple2<MatrixIndexes,MatrixBlock>(ixOut, blkOut);
			}			
			
		}
	}
	
	/**
	 * 
	 */
	private static class RDDQuaternaryFunction2 extends RDDQuaternaryBaseFunction //two rdd input
		implements PairFunction<Tuple2<MatrixIndexes, Tuple2<MatrixBlock,MatrixBlock>>, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = 7493974462943080693L;
		
		public RDDQuaternaryFunction2( QuaternaryOperator qop, PartitionedBroadcast<MatrixBlock> bcU, PartitionedBroadcast<MatrixBlock> bcV ) 
			throws DMLRuntimeException
		{
			super(qop, bcU, bcV);
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>> arg0)
			throws Exception 
		{
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
			return new Tuple2<MatrixIndexes,MatrixBlock>(ixOut, blkOut);
		}
	}
	
	/**
	 * 
	 */
	private static class RDDQuaternaryFunction3 extends RDDQuaternaryBaseFunction //three rdd input
		implements PairFunction<Tuple2<MatrixIndexes, Tuple2<Tuple2<MatrixBlock,MatrixBlock>,MatrixBlock>>, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = -2294086455843773095L;
		
		public RDDQuaternaryFunction3( QuaternaryOperator qop, PartitionedBroadcast<MatrixBlock> bcU, PartitionedBroadcast<MatrixBlock> bcV ) 
			throws DMLRuntimeException
		{
			super(qop, bcU, bcV);
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Tuple2<Tuple2<MatrixBlock, MatrixBlock>, MatrixBlock>> arg0)
			throws Exception 
		{
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
			return new Tuple2<MatrixIndexes,MatrixBlock>(ixOut, blkOut);
		}
	}
	
	/**
	 * Note: never called for wsigmoid/wdivmm (only wsloss)
	 */
	private static class RDDQuaternaryFunction4 extends RDDQuaternaryBaseFunction //four rdd input
		implements PairFunction<Tuple2<MatrixIndexes,Tuple2<Tuple2<Tuple2<MatrixBlock,MatrixBlock>,MatrixBlock>,MatrixBlock>>,MatrixIndexes,MatrixBlock>
	{
		private static final long serialVersionUID = 7328911771600289250L;
		
		public RDDQuaternaryFunction4( QuaternaryOperator qop ) 
			throws DMLRuntimeException
		{ 
			super(qop, null, null);		
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Tuple2<Tuple2<Tuple2<MatrixBlock, MatrixBlock>, MatrixBlock>, MatrixBlock>> arg0)
			throws Exception 
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
			return new Tuple2<MatrixIndexes,MatrixBlock>(ixOut, blkOut);
		}
	}
	
	private static class TransposeFactorIndexesFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -2571724736131823708L;
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();

			//swap the matrix indexes
			MatrixIndexes ixOut = new MatrixIndexes(ixIn.getColumnIndex(), ixIn.getRowIndex());
			MatrixBlock blkOut = new MatrixBlock(blkIn);
			
			//output new tuple
			return new Tuple2<MatrixIndexes, MatrixBlock>(ixOut,blkOut);
		}
		
	}

	private static class ReplicateBlocksFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -1184696764516975609L;
		
		private long _len = -1;
		private long _blen = -1;
		private boolean _left = false;
		
		public ReplicateBlocksFunction(long len, long blen, boolean left)
		{
			_len = len;
			_blen = blen;
			_left = left;
		}
		
		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			LinkedList<Tuple2<MatrixIndexes, MatrixBlock>> ret = new LinkedList<Tuple2<MatrixIndexes, MatrixBlock>>();
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();
			
			long numBlocks = (long) Math.ceil((double)_len/_blen); 
			
			if( _left ) //LHS MATRIX
			{
				//replicate wrt # column blocks in RHS
				long i = ixIn.getRowIndex();
				for( long j=1; j<=numBlocks; j++ ) {
					MatrixIndexes tmpix = new MatrixIndexes(i, j);
					MatrixBlock tmpblk = new MatrixBlock(blkIn);
					ret.add( new Tuple2<MatrixIndexes, MatrixBlock>(tmpix, tmpblk) );
				}
			} 
			else // RHS MATRIX
			{
				//replicate wrt # row blocks in LHS
				long j = ixIn.getColumnIndex();
				for( long i=1; i<=numBlocks; i++ ) {
					MatrixIndexes tmpix = new MatrixIndexes(i, j);
					MatrixBlock tmpblk = new MatrixBlock(blkIn);
					ret.add( new Tuple2<MatrixIndexes, MatrixBlock>(tmpix, tmpblk) );
				}
			}
			
			//output list of new tuples
			return ret;
		}
	}
}
