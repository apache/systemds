/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;


import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.lops.MapMultChain;
import com.ibm.bi.dml.lops.MapMultChain.ChainType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.instructions.spark.functions.RDDAggregateUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

/**
 * 
 */
public class MapmmChainSPInstruction extends SPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private ChainType _chainType = null;
	
	private CPOperand _input1 = null;
	private CPOperand _input2 = null;
	private CPOperand _input3 = null;	
	private CPOperand _output = null;	
	
	
	public MapmmChainSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, 
			                       ChainType type, String opcode, String istr )
	{
		super(op, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.MAPMMCHAIN;
		
		_input1 = in1;
		_input2 = in2;
		_output = out;
		
		_chainType = type;
	}
	
	public MapmmChainSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, 
                                   ChainType type, String opcode, String istr )
	{
		super(op, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.MAPMMCHAIN;
		
		_input1 = in1;
		_input2 = in2;
		_input3 = in3;
		_output = out;
		
		_chainType = type;
	}

	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MapmmChainSPInstruction parseInstruction( String str ) 
		throws DMLRuntimeException 
	{
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in3 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String opcode = InstructionUtils.getOpCode(str);

		//check supported opcode 
		if ( !opcode.equalsIgnoreCase(MapMultChain.OPCODE)){
			throw new DMLRuntimeException("MapmmChainSPInstruction.parseInstruction():: Unknown opcode " + opcode);	
		}
		
		//check number of fields (2/3 inputs, output, type)
		InstructionUtils.checkNumFields ( str, 4, 5 );
			
		//parse instruction parts (without exec type)
		String[] parts = InstructionUtils.getInstructionPartsWithValueType( str );	
		in1.split(parts[1]);
		in2.split(parts[2]);
		
		if( parts.length==5 )
		{
			out.split(parts[3]);
			ChainType type = ChainType.valueOf(parts[4]);
			
			return new MapmmChainSPInstruction(null, in1, in2, out, type, opcode, str);
		}
		else //parts.length==6
		{
			in3.split(parts[3]);
			out.split(parts[4]);
			ChainType type = ChainType.valueOf(parts[5]);
		
			return new MapmmChainSPInstruction(null, in1, in2, in3, out, type, opcode, str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{	
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get rdd and broadcast inputs
		JavaPairRDD<MatrixIndexes,MatrixBlock> inX = sec.getBinaryBlockRDDHandleForVariable( _input1.getName() );
		Broadcast<PartitionedMatrixBlock> inV = sec.getBroadcastForVariable( _input2.getName() );
		Broadcast<PartitionedMatrixBlock> inW = (_chainType==ChainType.XtwXv) ? sec.getBroadcastForVariable( _input3.getName() ) : null;
		
		//execute mapmmchain (guaranteed to have single output block)
		RDDMapMMChainFunction fmmc = new RDDMapMMChainFunction(_chainType, inV, inW);
		JavaPairRDD<MatrixIndexes,MatrixBlock> tmp = inX.mapToPair(fmmc);
		MatrixBlock out = RDDAggregateUtils.sumStable(tmp);
		
		//put output block into symbol table (no lineage because single block)
		//this also includes implicit maintenance of matrix characteristics
		sec.setMatrixOutput(_output.getName(), out);
	}
	
	/**
	 * 
	 * 
	 */
	private static class RDDMapMMChainFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 8197406787010296291L;

		private ChainType _type = null;
		
		private Broadcast<PartitionedMatrixBlock> _pmV = null;
		private Broadcast<PartitionedMatrixBlock> _pmW = null;
		
		public RDDMapMMChainFunction( ChainType type, Broadcast<PartitionedMatrixBlock> bV, Broadcast<PartitionedMatrixBlock> bW) 
			throws DMLRuntimeException, DMLUnsupportedOperationException
		{
			_type = type;
			
			//get first broadcast vector (always single block)
			_pmV = bV;
			
			//get second broadcast vector (partitioning for fast in memory lookup)
			if( _type == ChainType.XtwXv )
			{
				//in-memory rowblock partitioning (according to bclen of rdd)
				_pmW = bW;
			}
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			MatrixBlock pmV = _pmV.value().getMatrixBlock(1, 1);
			
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();
			int rowIx = (int)ixIn.getRowIndex();
			
			MatrixIndexes ixOut = new MatrixIndexes(1,1);
			MatrixBlock blkOut = new MatrixBlock();
			
			//core mapmmchain operation
			if( _type == ChainType.XtXv )
				blkIn.chainMatrixMultOperations(pmV, null, blkOut, _type);
			else if ( _type == ChainType.XtwXv )
			{
				PartitionedMatrixBlock pmW = _pmW.value();
				blkIn.chainMatrixMultOperations(pmV, pmW.getMatrixBlock(rowIx,1), blkOut, _type);
			}
				
			//output new tuple
			return new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut);
		}
	}
}
