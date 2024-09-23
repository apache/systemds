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


import org.apache.commons.lang3.tuple.Pair;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.sysds.common.Types;
import org.apache.sysds.lops.MapMultChain;
import org.apache.sysds.lops.MapMultChain.ChainType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.lineage.LineageTraceable;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import scala.Tuple2;

public class MapmmChainSPInstruction extends SPInstruction implements LineageTraceable {
	private ChainType _chainType = null;
	public CPOperand input1 = null;
	public CPOperand input2 = null;
	public CPOperand input3 = null;
	public CPOperand output = null;

	private MapmmChainSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, ChainType type, String opcode, String istr) {
		super(SPType.MAPMMCHAIN, op, opcode, istr);
		input1 = in1;
		input2 = in2;
		output = out;
		_chainType = type;
	}

	private MapmmChainSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, ChainType type, String opcode, String istr) {
		super(SPType.MAPMMCHAIN, op, opcode, istr);
		input1 = in1;
		input2 = in2;
		input3 = in3;
		output = out;
		_chainType = type;
	}

	public static MapmmChainSPInstruction parseInstruction( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType( str );	
		InstructionUtils.checkNumFields ( parts, 4, 5 );
		String opcode = parts[0];

		//check supported opcode 
		if ( !opcode.equalsIgnoreCase(MapMultChain.OPCODE)){
			throw new DMLRuntimeException("MapmmChainSPInstruction.parseInstruction():: Unknown opcode " + opcode);	
		}
			
		//parse instruction parts (without exec type)
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		
		if( parts.length==5 )
		{
			CPOperand out = new CPOperand(parts[3]);
			ChainType type = ChainType.valueOf(parts[4]);
			
			return new MapmmChainSPInstruction(null, in1, in2, out, type, opcode, str);
		}
		else //parts.length==6
		{
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[4]);
			ChainType type = ChainType.valueOf(parts[5]);
		
			return new MapmmChainSPInstruction(null, in1, in2, in3, out, type, opcode, str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get rdd and broadcast inputs
		JavaPairRDD<MatrixIndexes,MatrixBlock> inX = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() );
		PartitionedBroadcast<MatrixBlock> inV = sec.getBroadcastForVariable( input2.getName() );
		
		//execute mapmmchain (guaranteed to have single output block)
		MatrixBlock out = null;
		if( _chainType == ChainType.XtXv ) {
			JavaRDD<MatrixBlock> tmp = inX.values().map(new RDDMapMMChainFunction(inV));
			out = RDDAggregateUtils.sumStable(tmp);
		}
		else { // ChainType.XtwXv / ChainType.XtXvy
			PartitionedBroadcast<MatrixBlock> inW = sec.getBroadcastForVariable( input3.getName() );
			JavaRDD<MatrixBlock> tmp = inX.map(new RDDMapMMChainFunction2(inV, inW, _chainType));
			out = RDDAggregateUtils.sumStable(tmp);
		}
		
		//put output block into symbol table (no lineage because single block)
		//this also includes implicit maintenance of matrix characteristics
		sec.setMatrixOutput(output.getName(), out);
	}

	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		CPOperand chainT = new CPOperand(_chainType.name(), Types.ValueType.INT64, Types.DataType.SCALAR, true);
		return Pair.of(output.getName(), new LineageItem(getOpcode(),
			LineageItemUtils.getLineage(ec, input1, input2, input3, chainT)));
	}

	/**
	 * This function implements the chain type XtXv which requires just one broadcast and
	 * no access to any indexes of matrix blocks.
	 * 
	 */
	private static class RDDMapMMChainFunction implements Function<MatrixBlock, MatrixBlock> 
	{
		private static final long serialVersionUID = 8197406787010296291L;

		private PartitionedBroadcast<MatrixBlock> _pmV = null;
		
		public RDDMapMMChainFunction( PartitionedBroadcast<MatrixBlock> bV) {
			//get first broadcast vector (always single block)
			_pmV = bV;
		}
		
		@Override
		public MatrixBlock call( MatrixBlock arg0 ) {
			MatrixBlock pmV = _pmV.getBlock(1, 1);
			//execute mapmmchain operation
			return arg0.chainMatrixMultOperations(pmV, 
					null, new MatrixBlock(), ChainType.XtXv);
		}
	}
	
	/**
	 * This function implements the chain type XtwXv which requires two broadcasts and
	 * access to the row index of a given matrix block. 
	 */
	private static class RDDMapMMChainFunction2 implements Function<Tuple2<MatrixIndexes, MatrixBlock>, MatrixBlock> 
	{
		private static final long serialVersionUID = -7926980450209760212L;

		private PartitionedBroadcast<MatrixBlock> _pmV = null;
		private PartitionedBroadcast<MatrixBlock> _pmW = null;
		private ChainType _chainType = null;
		
		public RDDMapMMChainFunction2( PartitionedBroadcast<MatrixBlock> bV, PartitionedBroadcast<MatrixBlock> bW, ChainType chain) {
			//get both broadcast vectors (first always single block)
			_pmV = bV;
			_pmW = bW;
			_chainType = chain;
		}
		
		@Override
		public MatrixBlock call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) {
			MatrixBlock pmV = _pmV.getBlock(1, 1);
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();
			int rowIx = (int)ixIn.getRowIndex();
			//execute mapmmchain operation
			return blkIn.chainMatrixMultOperations(pmV, 
					_pmW.getBlock(rowIx,1), new MatrixBlock(), _chainType);
		}
	}
}
