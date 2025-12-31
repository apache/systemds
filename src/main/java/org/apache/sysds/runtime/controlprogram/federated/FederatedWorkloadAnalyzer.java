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

package org.apache.sysds.runtime.controlprogram.federated;

import java.util.concurrent.ConcurrentHashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.cost.InstructionTypeCounter;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.IndexFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.Mean;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.AggregateBinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.AggregateUnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ComputationCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MMChainCPInstruction;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class FederatedWorkloadAnalyzer {
	protected static final Log LOG = LogFactory.getLog(FederatedWorkloadAnalyzer.class.getName());

	/** Frequency value for how many instructions before we do a pass for compression */
	private static final int compressRunFrequency = 10;

	/** Instruction maps to interesting variables */
	private final ConcurrentHashMap<Long, ConcurrentHashMap<Long, InstructionTypeCounter>> m;

	/** Counter to decide when to do a compress run */
	private int counter;

	public FederatedWorkloadAnalyzer() {
		m = new ConcurrentHashMap<>();
		counter = 0;
	}

	public void incrementWorkload(ExecutionContext ec, long tid, Instruction ins) {
		if(ins instanceof ComputationCPInstruction)
			incrementWorkload(ec, tid, (ComputationCPInstruction) ins);
		// currently we ignore everything that is not CP instructions
	}

	public void compressRun(ExecutionContext ec, long tid) {
		if(counter >= compressRunFrequency) {
			counter = 0;
			get(tid).forEach((K, V) -> CompressedMatrixBlockFactory.compressAsync(ec, Long.toString(K), V));
		}
	}

	private void incrementWorkload(ExecutionContext ec, long tid, ComputationCPInstruction cpIns) {
		incrementWorkload(ec, get(tid), cpIns);
	}

	public void incrementWorkload(ExecutionContext ec, ConcurrentHashMap<Long, InstructionTypeCounter> mm,
		ComputationCPInstruction cpIns) {
		// TODO: Count transitive closure via lineage
		// TODO: add more operations
		if(cpIns instanceof AggregateBinaryCPInstruction) {
			final String n1 = cpIns.input1.getName();
			MatrixObject d1 = (MatrixObject) ec.getCacheableData(n1);
			final String n2 = cpIns.input2.getName();
			MatrixObject d2 = (MatrixObject) ec.getCacheableData(n2);

			int r1 = (int) d1.getDim(0);
			int c1 = (int) d1.getDim(1);
			int r2 = (int) d2.getDim(0);
			int c2 = (int) d2.getDim(1);
			if(validSize(r1, c1)) {
				getOrMakeCounter(mm, Long.parseLong(n1)).incRMM(c2);
				// safety add overlapping decompress for RMM
				getOrMakeCounter(mm, Long.parseLong(n1)).incOverlappingDecompressions(c2);
				counter++;
			}
			if(validSize(r2, c2)) {
				getOrMakeCounter(mm, Long.parseLong(n2)).incLMM(r1);
				counter++;
			}
		}
		else if(cpIns instanceof MMChainCPInstruction) {
			final String n1 = cpIns.input1.getName();
			getOrMakeCounter(mm, Long.parseLong(n1)).incRMM(1);
			getOrMakeCounter(mm, Long.parseLong(n1)).incLMM(1);
			counter++;
		}
		else if(cpIns instanceof AggregateUnaryCPInstruction) {
			Operator op = cpIns.getOperator();
			final String n1 = cpIns.input1.getName();
			long id = Long.parseLong(n1);
			// MatrixObject d1 = (MatrixObject) ec.getCacheableData(n1);
			// int r1 = (int) d1.getDim(0);
			// int c1 = (int) d1.getDim(1);
			if(op instanceof AggregateUnaryOperator) {
				AggregateUnaryOperator aop = (AggregateUnaryOperator) op;
				IndexFunction idxF = aop.indexFn;
				getOrMakeCounter(mm, id).incDictOps();
				if(idxF instanceof ReduceCol) {
					if((aop.aggOp.increOp.fn instanceof KahanPlus //
						|| aop.aggOp.increOp.fn instanceof Plus //
						|| aop.aggOp.increOp.fn instanceof Mean)) {
						getOrMakeCounter(mm, id).incDictOps();
					}
					else {
						// increment decompression if row reduce.
						getOrMakeCounter(mm, id).incDecompressions();
					}
				}
				else {
					getOrMakeCounter(mm, id).incDictOps();
				}
			}
		}

	}

	private static InstructionTypeCounter getOrMakeCounter(ConcurrentHashMap<Long, InstructionTypeCounter> mm, long id) {
		if(mm.containsKey(id)) {
			return mm.get(id);
		}
		else {
			final InstructionTypeCounter r = new InstructionTypeCounter();
			mm.put(id, r);
			return r;
		}
	}

	private ConcurrentHashMap<Long, InstructionTypeCounter> get(long tid) {
		if(m.containsKey(tid))
			return m.get(tid);
		else {
			final ConcurrentHashMap<Long, InstructionTypeCounter> r = new ConcurrentHashMap<>();
			m.put(tid, r);
			return r;
		}
	}

	private static boolean validSize(int nRow, int nCol) {
		return nRow > 90 && nRow >= nCol;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("  Counter: ");
		sb.append(counter);
		sb.append("\n");
		sb.append(m);

		return sb.toString();
	}
}
