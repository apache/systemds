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

import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.storage.RDDInfo;
import org.apache.spark.storage.StorageLevel;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.SingletonLookupHashMap;
import org.apache.sysds.runtime.compress.cost.CostEstimatorBuilder;
import org.apache.sysds.runtime.compress.cost.CostEstimatorFactory.CostType;
import org.apache.sysds.runtime.compress.workload.WTreeRoot;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;

import scala.Tuple2;

public class CompressionSPInstruction extends UnarySPInstruction {
	private static final Log LOG = LogFactory.getLog(CompressionSPInstruction.class.getName());

	private final int _singletonLookupID;

	private CompressionSPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr,
		int singletonLookupID) {
		super(SPType.Compression, op, in, out, opcode, istr);
		_singletonLookupID = singletonLookupID;
	}

	public static CompressionSPInstruction parseInstruction(String str) {
		InstructionUtils.checkNumFields(str, 2, 3);
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);

		if(parts.length == 4) {
			int treeNodeID = Integer.parseInt(parts[3]);
			return new CompressionSPInstruction(null, in1, out, opcode, str, treeNodeID);
		}
		else
			return new CompressionSPInstruction(null, in1, out, opcode, str, 0);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext) ec;

		// get input rdd handle
		JavaPairRDD<MatrixIndexes, MatrixBlock> in = sec.getBinaryMatrixBlockRDDHandleForVariable(input1.getName());

		// construct the compression mapping function
		Function<MatrixBlock, MatrixBlock> mappingFunction;
		if(_singletonLookupID == 0)
			mappingFunction = new CompressionFunction();
		else {
			WTreeRoot root = (WTreeRoot) SingletonLookupHashMap.getMap().get(_singletonLookupID);
			CostEstimatorBuilder costBuilder = new CostEstimatorBuilder(root);
			mappingFunction = new CompressionWorkloadFunction(costBuilder);
		}

		// execute compression
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = in.mapValues(mappingFunction);
		if(LOG.isTraceEnabled()) {
			in.persist(StorageLevel.MEMORY_AND_DISK());
			out.persist(StorageLevel.MEMORY_AND_DISK());
			long sparkSizeIn = 0;
			long sparkSizeOut = 0;
			long blockSizesIn = reduceSizes(in.mapValues(new SizeFunction()).collect());
			long blockSizesOut = reduceSizes(out.mapValues(new SizeFunction()).collect());
			for(RDDInfo info : sec.getSparkContext().sc().getRDDStorageInfo()) {
				if(info.id() == out.id())
					sparkSizeOut = info.memSize();
				else if(info.id() == in.id())
					sparkSizeIn = info.memSize();
			}
			StringBuilder sb = new StringBuilder();
			sb.append("Spark Compression Instruction sizes:");
			sb.append(String.format("\nSBCompress: InSize:       %16d", sparkSizeIn));
			sb.append(String.format("\nSBCompress: InBlockSize:  %16d", blockSizesIn));
			sb.append(String.format("\nSBCompress: OutSize:      %16d", sparkSizeOut));
			sb.append(String.format("\nSBCompress: OutBlockSize: %16d", blockSizesOut));
			LOG.trace(sb.toString());
		}

		// set outputs
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(input1.getName(), output.getName());
	}

	public static class CompressionFunction implements Function<MatrixBlock, MatrixBlock> {
		private static final long serialVersionUID = -6528833083609423922L;

		@Override
		public MatrixBlock call(MatrixBlock arg0) throws Exception {
			CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setIsInSparkInstruction()
				.setCostType(CostType.MEMORY);
			return CompressedMatrixBlockFactory.compress(arg0, csb).getLeft();
		}
	}

	public static class CompressionWorkloadFunction implements Function<MatrixBlock, MatrixBlock> {
		private static final long serialVersionUID = -65288330833922L;

		final CostEstimatorBuilder costBuilder;

		public CompressionWorkloadFunction(CostEstimatorBuilder costBuilder) {
			this.costBuilder = costBuilder;
		}

		@Override
		public MatrixBlock call(MatrixBlock arg0) throws Exception {
			CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setIsInSparkInstruction();
			return CompressedMatrixBlockFactory.compress(arg0, InfrastructureAnalyzer.getLocalParallelism(), csb, costBuilder)
				.getLeft();
		}
	}

	public static class SizeFunction implements Function<MatrixBlock, Long> {
		private static final long serialVersionUID = 1L;

		public SizeFunction() {

		}

		@Override
		public Long call(MatrixBlock arg0) throws Exception {
			return arg0.getInMemorySize();
		}
	}

	public static Long reduceSizes(List<Tuple2<MatrixIndexes, Long>> in) {
		long sum = 0;
		for(Tuple2<MatrixIndexes, Long> e : in)
			sum += e._2();
		return sum;
	}
}
