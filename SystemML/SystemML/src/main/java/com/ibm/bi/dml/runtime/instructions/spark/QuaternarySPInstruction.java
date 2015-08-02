/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.lops.WeightedSigmoid;
import com.ibm.bi.dml.lops.WeightedSigmoidR;
import com.ibm.bi.dml.lops.WeightedSquaredLoss;
import com.ibm.bi.dml.lops.WeightedSquaredLossR;
import com.ibm.bi.dml.lops.WeightedSigmoid.WSigmoidType;
import com.ibm.bi.dml.lops.WeightedSquaredLoss.WeightsType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.cp.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.instructions.spark.utils.RDDAggregateUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.QuaternaryOperator;

/**
 * 
 */
public class QuaternarySPInstruction extends ComputationSPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
		String opcode = InstructionUtils.getOpCode(str);
		
		//validity check
		if (   !WeightedSquaredLoss.OPCODE.equalsIgnoreCase(opcode)   //mapwsloss
			&& !WeightedSquaredLossR.OPCODE.equalsIgnoreCase(opcode)  //redwsloss
			&& !WeightedSigmoid.OPCODE.equalsIgnoreCase(opcode)    //mapwsigmoid
			&& !WeightedSigmoidR.OPCODE.equalsIgnoreCase(opcode) ) //redwsigmoid
		{
			throw new DMLRuntimeException("Quaternary.parseInstruction():: Unknown opcode " + opcode);
		}
		
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in3 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in4 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		
		//instruction parsing
		if(    WeightedSquaredLoss.OPCODE.equalsIgnoreCase(opcode)    //wsloss
			|| WeightedSquaredLossR.OPCODE.equalsIgnoreCase(opcode) )
		{
			boolean isRed = WeightedSquaredLossR.OPCODE.equalsIgnoreCase(opcode);
			
			//check number of fields (4 inputs, output, type)
			if( isRed )
				InstructionUtils.checkNumFields ( str, 8 );
			else
				InstructionUtils.checkNumFields ( str, 6 );
				
			//parse instruction parts (without exec type)
			String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
			
			in1.split(parts[1]);
			in2.split(parts[2]);
			in3.split(parts[3]);
			in4.split(parts[4]);
			out.split(parts[5]);
			WeightsType wtype = WeightsType.valueOf(parts[6]);
			
			//in mappers always through distcache, in reducers through distcache/shuffle
			boolean cacheU = isRed ? Boolean.parseBoolean(parts[7]) : true;
			boolean cacheV = isRed ? Boolean.parseBoolean(parts[8]) : true;
			
			return new QuaternarySPInstruction(new QuaternaryOperator(wtype), in1, in2, in3, in4, out, cacheU, cacheV, opcode, str);	
		}
		else //wsigmoid
		{
			boolean isRed = WeightedSigmoidR.OPCODE.equalsIgnoreCase(opcode);
			
			//check number of fields (3 inputs, output, type)
			if( isRed )
				InstructionUtils.checkNumFields ( str, 7 );
			else
				InstructionUtils.checkNumFields ( str, 5 );
				
			//parse instruction parts (without exec type)
			String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
			
			in1.split(parts[1]);
			in2.split(parts[2]);
			in3.split(parts[3]);
			out.split(parts[4]);
			WSigmoidType wtype = WSigmoidType.valueOf(parts[5]);
			
			//in mappers always through distcache, in reducers through distcache/shuffle
			boolean cacheU = isRed ? Boolean.parseBoolean(parts[6]) : true;
			boolean cacheV = isRed ? Boolean.parseBoolean(parts[7]) : true;
			
			return new QuaternarySPInstruction(new QuaternaryOperator(wtype), in1, in2, in3, null, out, cacheU, cacheV, opcode, str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
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
		
		//map-side only operation (one rdd input, two broadcasts)
		if(    WeightedSquaredLoss.OPCODE.equalsIgnoreCase(getOpcode())  
			|| WeightedSigmoid.OPCODE.equalsIgnoreCase(getOpcode())     ) 
		{
			Broadcast<PartitionedMatrixBlock> bc1 = sec.getBroadcastForVariable( input2.getName() );
			Broadcast<PartitionedMatrixBlock> bc2 = sec.getBroadcastForVariable( input3.getName() );
			
			out = in.mapToPair(new RDDQuaternaryFunction1(qop, bc1, bc2));
			
			rddVars.add( input1.getName() );
			bcVars.add( input2.getName() );
			bcVars.add( input3.getName() );
		}
		//reduce-side operation (two/three/four rdd inputs, zero/one/two broadcasts)
		else 
		{
			Broadcast<PartitionedMatrixBlock> bc1 = _cacheU ? sec.getBroadcastForVariable( input2.getName() ) : null;
			Broadcast<PartitionedMatrixBlock> bc2 = _cacheV ? sec.getBroadcastForVariable( input3.getName() ) : null;
			JavaPairRDD<MatrixIndexes,MatrixBlock> inU = (!_cacheU) ? sec.getBinaryBlockRDDHandleForVariable( input2.getName() ) : null;
			JavaPairRDD<MatrixIndexes,MatrixBlock> inV = (!_cacheV) ? sec.getBinaryBlockRDDHandleForVariable( input3.getName() ) : null;
			JavaPairRDD<MatrixIndexes,MatrixBlock> inW = (qop.wtype1!=null && qop.wtype1!=WeightsType.NONE) ? 
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
				        .flatMapToPair(new RDDQuaternaryFunction2(qop, bc1, bc2));
			else if( inU == null && inV != null && inW == null )
				out = in.join(inV)
				        .flatMapToPair(new RDDQuaternaryFunction2(qop, bc1, bc2));
			else if( inU == null && inV == null && inW != null )
				out = in.join(inW)
				        .flatMapToPair(new RDDQuaternaryFunction2(qop, bc1, bc2));
			//function calls w/ three rdd inputs
			else if( inU != null && inV != null && inW == null )
				out = in.join(inU).join(inV)
				        .flatMapToPair(new RDDQuaternaryFunction3(qop, bc1, bc2));
			else if( inU != null && inV == null && inW != null )
				out = in.join(inU).join(inW)
				        .flatMapToPair(new RDDQuaternaryFunction3(qop, bc1, bc2));
			else if( inU == null && inV != null && inW != null )
				out = in.join(inV).join(inW)
				        .flatMapToPair(new RDDQuaternaryFunction3(qop, bc1, bc2));
			//function call w/ four rdd inputs
			else 
				out = in.join(inU).join(inV).join(inW)
				        .flatMapToPair(new RDDQuaternaryFunction4(qop));
			
			//keep variable names for lineage maintenance
			if( inU == null ) bcVars.add(input2.getName()); else rddVars.add(input2.getName());
			if( inV == null ) bcVars.add(input3.getName()); else rddVars.add(input3.getName());
			if( inW != null ) rddVars.add(_input4.getName());
		}
		
		//output handling
		if( qop.wtype1 != null ) //mapwsloss, redwsloss
		{
			//full aggregate and cast to scalar
			MatrixBlock tmp = RDDAggregateUtils.sumStable(out);
			DoubleObject ret = new DoubleObject(tmp.getValue(0, 0));
			sec.setVariable(output.getName(), ret);
		}
		else //mapwsigmoid, redwsigmoid
		{
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			//maintain lineage information for output rdd
			for( String rddVar : rddVars )
				sec.addLineageRDD(output.getName(), rddVar);
			for( String bcVar : bcVars )
				sec.addLineageBroadcast(output.getName(), bcVar);
		}
	}
	
	private static class RDDQuaternaryFunction1 //one rdd input
		implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = -8209188316939435099L;
		
		private QuaternaryOperator _qop = null;
		private Broadcast<PartitionedMatrixBlock> _pmU = null;
		private Broadcast<PartitionedMatrixBlock> _pmV = null;
		
		public RDDQuaternaryFunction1( QuaternaryOperator qop, Broadcast<PartitionedMatrixBlock> bcU, Broadcast<PartitionedMatrixBlock> bcV ) 
			throws DMLRuntimeException, DMLUnsupportedOperationException
		{
			_qop = qop;		
			
			//get the broadcast matrices and partition on demand
			_pmU = bcU;
			_pmV = bcV;
		}
	
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
			throws Exception 
		{
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();
			MatrixBlock blkOut = new MatrixBlock();
			
			MatrixBlock mbU = _pmU.value().getMatrixBlock((int)ixIn.getRowIndex(), 1);
			MatrixBlock mbV = _pmV.value().getMatrixBlock((int)ixIn.getColumnIndex(), 1);
			
			//execute core operation
			blkIn.quaternaryOperations(_qop, mbU, mbV, null, blkOut);
			
			//create return tuple
			return new Tuple2<MatrixIndexes,MatrixBlock>(new MatrixIndexes(ixIn), blkOut);
		}
	}
	
	private static class RDDQuaternaryFunction2 //two rdd input
		implements PairFlatMapFunction<Tuple2<MatrixIndexes, Tuple2<MatrixBlock,MatrixBlock>>, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = 7493974462943080693L;
		
		private QuaternaryOperator _qop = null;
		private Broadcast<PartitionedMatrixBlock> _pmU = null;
		private Broadcast<PartitionedMatrixBlock> _pmV = null;
		
		public RDDQuaternaryFunction2( QuaternaryOperator qop, Broadcast<PartitionedMatrixBlock> bcU, Broadcast<PartitionedMatrixBlock> bcV ) 
			throws DMLRuntimeException, DMLUnsupportedOperationException
		{
			_qop = qop;		
			
			//get the broadcast matrices and partition on demand
			_pmU = bcU;
			_pmV = bcV;
		}

		@Override
		@SuppressWarnings("unchecked")
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>> arg0)
			throws Exception 
		{
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn1 = arg0._2()._1();
			MatrixBlock blkIn2 = arg0._2()._2();
			MatrixBlock blkOut = new MatrixBlock();
			
			MatrixBlock mbU = (_pmU!=null)?_pmU.value().getMatrixBlock((int)ixIn.getRowIndex(), 1) : blkIn2;
			MatrixBlock mbV = (_pmV!=null)?_pmV.value().getMatrixBlock((int)ixIn.getColumnIndex(), 1) : blkIn2;
			MatrixBlock mbW = (_qop.wtype1!=null&&_qop.wtype1!=WeightsType.NONE)? blkIn2 : null;
			
			//execute core operation
			blkIn1.quaternaryOperations(_qop, mbU, mbV, mbW, blkOut);
			
			//create return tuple
			return Arrays.asList(new Tuple2<MatrixIndexes,MatrixBlock>(new MatrixIndexes(ixIn), blkOut));
		}
	}
	
	private static class RDDQuaternaryFunction3 //three rdd input
		implements PairFlatMapFunction<Tuple2<MatrixIndexes, Tuple2<Tuple2<MatrixBlock,MatrixBlock>,MatrixBlock>>, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = -2294086455843773095L;
		
		private QuaternaryOperator _qop = null;
		private Broadcast<PartitionedMatrixBlock> _pmU = null;
		private Broadcast<PartitionedMatrixBlock> _pmV = null;
		
		public RDDQuaternaryFunction3( QuaternaryOperator qop, Broadcast<PartitionedMatrixBlock> bcU, Broadcast<PartitionedMatrixBlock> bcV ) 
			throws DMLRuntimeException, DMLUnsupportedOperationException
		{
			_qop = qop;		
			
			//get the broadcast matrices and partition on demand
			_pmU = bcU;
			_pmV = bcV;
		}

		@Override
		@SuppressWarnings("unchecked")
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, Tuple2<Tuple2<MatrixBlock, MatrixBlock>, MatrixBlock>> arg0)
			throws Exception 
		{
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn1 = arg0._2()._1()._1();
			MatrixBlock blkIn2 = arg0._2()._1()._2();
			MatrixBlock blkIn3 = arg0._2()._2();
			
			MatrixBlock blkOut = new MatrixBlock();
			
			MatrixBlock mbU = (_pmU!=null)?_pmU.value().getMatrixBlock((int)ixIn.getRowIndex(), 1) : blkIn2;
			MatrixBlock mbV = (_pmV!=null)?_pmV.value().getMatrixBlock((int)ixIn.getColumnIndex(), 1) : 
				              (_pmU!=null)? blkIn2 : blkIn3;
			MatrixBlock mbW = (_qop.wtype1!=null&&_qop.wtype1!=WeightsType.NONE)? blkIn3 : null;
			
			//execute core operation
			blkIn1.quaternaryOperations(_qop, mbU, mbV, mbW, blkOut);
			
			//create return tuple
			return Arrays.asList(new Tuple2<MatrixIndexes,MatrixBlock>(new MatrixIndexes(ixIn), blkOut));
		}
	}
	
	private static class RDDQuaternaryFunction4 //four rdd input
		implements PairFlatMapFunction<Tuple2<MatrixIndexes, Tuple2<Tuple2<Tuple2<MatrixBlock,MatrixBlock>,MatrixBlock>,MatrixBlock>>, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = 7328911771600289250L;
		
		private QuaternaryOperator _qop = null;
		
		public RDDQuaternaryFunction4( QuaternaryOperator qop ) 
			throws DMLRuntimeException, DMLUnsupportedOperationException
		{
			_qop = qop;			
		}
	
		@Override
		@SuppressWarnings("unchecked")
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, Tuple2<Tuple2<Tuple2<MatrixBlock, MatrixBlock>, MatrixBlock>, MatrixBlock>> arg0)
			throws Exception 
		{
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn1 = arg0._2()._1()._1()._1();
			MatrixBlock mbU = arg0._2()._1()._1()._2();
			MatrixBlock mbV = arg0._2()._1()._2();
			MatrixBlock mbW = arg0._2()._2();
			MatrixBlock blkOut = new MatrixBlock();
			
			//execute core operation
			blkIn1.quaternaryOperations(_qop, mbU, mbV, mbW, blkOut);
			
			//create return tuple
			return Arrays.asList(new Tuple2<MatrixIndexes,MatrixBlock>(new MatrixIndexes(ixIn), blkOut));
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
