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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

import scala.Tuple2;

public class ZipmmSPInstruction extends BinarySPInstruction {
	// internal flag to apply left-transpose rewrite or not
	private boolean _tRewrite = true;

	private ZipmmSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, boolean tRewrite,
			String opcode, String istr) {
		super(SPType.ZIPMM, op, in1, in2, out, opcode, istr);
		_tRewrite = tRewrite;
	}

	public static ZipmmSPInstruction parseInstruction( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if ( opcode.equalsIgnoreCase(Opcodes.ZIPMM.toString())) {
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			boolean tRewrite = Boolean.parseBoolean(parts[4]);
			AggregateBinaryOperator aggbin = InstructionUtils.getMatMultOperator(1);
			
			return new ZipmmSPInstruction(aggbin, in1, in2, out, tRewrite, opcode, str);
		} 
		else {
			throw new DMLRuntimeException("ZipmmSPInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get rdd inputs (for computing r = t(X)%*%y via r = t(t(y)%*%X))
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() ); //X
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryMatrixBlockRDDHandleForVariable( input2.getName() ); //y

		if (ConfigurationManager.isMaxPrallelizeEnabled()) {
			try {
					ZipmmTask task = new ZipmmTask(in1, in2, _tRewrite);
				Future<MatrixBlock> future_out = CommonThreadPool.getDynamicPool().submit(task);
				LineageItem li = !LineageCacheConfig.ReuseCacheType.isNone() ? getLineageItem(ec).getValue() : null;
				sec.setMatrixOutputAndLineage(output.getName(), future_out, li);
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}
		}
		else {
			//process core zipmm matrix multiply (in contrast to cpmm, the join over original indexes
			//preserves the original partitioning and with that potentially unnecessary join shuffle)
			JavaRDD<MatrixBlock> out = in1.join(in2).values()     // join over original indexes
				.map(new ZipMultiplyFunction(_tRewrite));  // compute block multiplications, incl t(y)

			//single-block aggregation (guaranteed by zipmm blocksize constraint)
			MatrixBlock out2 = RDDAggregateUtils.sumStable(out);

			//final transpose of result (for t(t(y)%*%X))), if transpose rewrite
			if(_tRewrite) {
				ReorgOperator rop = new ReorgOperator(SwapIndex.getSwapIndexFnObject());
				out2 = out2.reorgOperations(rop, new MatrixBlock(), 0, 0, 0);
			}

			//put output block into symbol table (no lineage because single block)
			//this also includes implicit maintenance of matrix characteristics
			sec.setMatrixOutput(output.getName(), out2);
		}
	}

	private static class ZipMultiplyFunction implements Function<Tuple2<MatrixBlock,MatrixBlock>, MatrixBlock> 
	{
		private static final long serialVersionUID = -6669267794926220287L;
		
		private AggregateBinaryOperator _abop = null;
		private ReorgOperator _rop = null;
		private boolean _tRewrite = true;
		
		public ZipMultiplyFunction(boolean tRewrite)
		{
			_tRewrite = tRewrite;
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			_abop = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
			_rop = new ReorgOperator(SwapIndex.getSwapIndexFnObject());
		}

		@Override
		public MatrixBlock call(Tuple2<MatrixBlock, MatrixBlock> arg0)
			throws Exception 
		{
			MatrixBlock in1 = _tRewrite ? arg0._1() : arg0._2();
			MatrixBlock in2 = _tRewrite ? arg0._2() : arg0._1();
			
			//transpose right input (for vectors no-op)
			MatrixBlock tmp = in2.reorgOperations(_rop, new MatrixBlock(), 0, 0, 0);
			
			//core matrix multiplication (for t(y)%*%X or t(X)%*%y)
			return OperationsOnMatrixValues.matMult(tmp, in1, new MatrixBlock(), _abop);
		}
	}

	private static class ZipmmTask implements Callable<MatrixBlock> {
		JavaPairRDD<MatrixIndexes, MatrixBlock> _in1;
		JavaPairRDD<MatrixIndexes, MatrixBlock> _in2;
		boolean _tRewrite;

		ZipmmTask(JavaPairRDD<MatrixIndexes, MatrixBlock> in1, JavaPairRDD<MatrixIndexes, MatrixBlock> in2, boolean tRw) {
			_in1 = in1;
			_in2 = in2;
			_tRewrite = tRw;
		}
		@Override
		public MatrixBlock call() {
			//process core zipmm matrix multiply (in contrast to cpmm, the join over original indexes
			//preserves the original partitioning and with that potentially unnecessary join shuffle)
			JavaRDD<MatrixBlock> out = _in1.join(_in2).values()     // join over original indexes
				.map(new ZipMultiplyFunction(_tRewrite));  // compute block multiplications, incl t(y)

			//single-block aggregation (guaranteed by zipmm blocksize constraint)
			MatrixBlock out2 = RDDAggregateUtils.sumStable(out);

			//final transpose of result (for t(t(y)%*%X))), if transpose rewrite
			if( _tRewrite ) {
				ReorgOperator rop = new ReorgOperator(SwapIndex.getSwapIndexFnObject());
				out2 = out2.reorgOperations(rop, new MatrixBlock(), 0, 0, 0);
			}

			return out2;
		}
	}
}
