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


import java.util.concurrent.Callable;
import java.util.concurrent.Future;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.util.CommonThreadPool;

import scala.Tuple2;

public class TsmmSPInstruction extends UnarySPInstruction {
	private MMTSJType _type = null;

	private TsmmSPInstruction(Operator op, CPOperand in1, CPOperand out, MMTSJType type, String opcode, String istr) {
		super(SPType.TSMM, op, in1, out, opcode, istr);
		_type = type;
	}

	public static TsmmSPInstruction parseInstruction( String str ) {
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		//check supported opcode 
		if ( !opcode.equalsIgnoreCase(Opcodes.TSMM.toString()) )
			throw new DMLRuntimeException("TsmmSPInstruction.parseInstruction():: Unknown opcode " + opcode);
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		MMTSJType type = MMTSJType.valueOf(parts[3]);
		return new TsmmSPInstruction(null, in1, out, type, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() );

		if (ConfigurationManager.isMaxPrallelizeEnabled()) {
			try {
				TsmmTask task = new TsmmTask(in, _type);
				Future<MatrixBlock> future_out = CommonThreadPool.getDynamicPool().submit(task);
				LineageItem li = !LineageCacheConfig.ReuseCacheType.isNone() ? getLineageItem(ec).getValue() : null;
				sec.setMatrixOutputAndLineage(output.getName(), future_out, li);
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}
		}
		else {
			//execute tsmm instruction (always produce exactly one output block)
			//(this formulation with values() requires --conf spark.driver.maxResultSize=0)
			JavaRDD<MatrixBlock> tmp = in.map(new RDDTSMMFunction(_type));
			MatrixBlock out = RDDAggregateUtils.sumStable(tmp);

			//put output block into symbol table (no lineage because single block)
			//this also includes implicit maintenance of matrix characteristics
			sec.setMatrixOutput(output.getName(), out);
		}
	}

	public MMTSJType getMMTSJType()
	{
		return _type;
	}

	private static class RDDTSMMFunction implements Function<Tuple2<MatrixIndexes,MatrixBlock>, MatrixBlock> 
	{
		private static final long serialVersionUID = 2935770425858019666L;
		
		private MMTSJType _type = null;
		
		public RDDTSMMFunction( MMTSJType type ) {
			_type = type;
		}
		
		@Override
		public MatrixBlock call( Tuple2<MatrixIndexes,MatrixBlock> arg0 ) 
			throws Exception 
		{
			//execute transpose-self matrix multiplication
			return arg0._2().transposeSelfMatrixMultOperations(new MatrixBlock(), _type);
		}
	}

	private static class TsmmTask implements Callable<MatrixBlock> {
		JavaPairRDD<MatrixIndexes, MatrixBlock> _in;
		MMTSJType _type;

		TsmmTask(JavaPairRDD<MatrixIndexes, MatrixBlock> in, MMTSJType type) {
			_in = in;
			_type = type;
		}
		@Override
		public MatrixBlock call() {
			//execute tsmm instruction (always produce exactly one output block)
			//(this formulation with values() requires --conf spark.driver.maxResultSize=0)
			JavaRDD<MatrixBlock> tmp = _in.map(new RDDTSMMFunction(_type));
			return RDDAggregateUtils.sumStable(tmp);
		}
	}
	
}
