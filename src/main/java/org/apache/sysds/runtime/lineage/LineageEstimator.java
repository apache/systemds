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

package org.apache.sysds.runtime.lineage;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import org.apache.commons.lang3.tuple.MutableTriple;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.ComputationCPInstruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

public class LineageEstimator 
{
	private static final Map<LineageItem, LineageEstimatorEntry> _cache = new HashMap<>();
	private static final Map<String, MutableTriple<String, Long, Double>> _savedPerOP = new HashMap<>();
	protected static long _startTimestamp;
	protected static long _cachesize = 0;
	protected static long _cacheFullCount = 0;
	protected static long _totReusableSize = 0;
	protected static long _totReusedSize = 0;
	private static final double CACHE_FRAC = 0.05; //5% of JVM
	protected static long CACHE_LIMIT;
	private static Comparator<MutableTriple<String, Long, Double>> savedOPComparator = (op1, op2) -> {
		return op1.getRight() == op2.getRight() ? 0 : op1.getRight() < op2.getRight() ? 1 : -1;  
	};
	protected static PriorityQueue<MutableTriple<String, Long, Double>> computeSavingInst = new PriorityQueue<>(savedOPComparator);
	
	static {
		long maxMem = InfrastructureAnalyzer.getLocalMaxMemory();
		CACHE_LIMIT = ((long)(CACHE_FRAC * maxMem));
		_startTimestamp = System.currentTimeMillis();
	}
	// Collected statistics descriptions:
	// _cache =            Captures all lineage traceable CP instructions and functions
	// computeSavingInst = A priority queue which captures reuse count and saved computetime
	//                     group by opcodes, order by saved computetime
	// _cachesize =        Total size of all entries saved in _cache
	// _totReusableSize =  Total size of all intermediates which are reusable as per
	//                     original lineage caching logic.
	// _totReusedSize =    Total size of all intermediates which are reused and 
	//                     are reusable as per original lineage caching logic.
	// _cacheFullCount =   Number of lineage traceable instructions since the beginning
	//                     of execution that fill up the cache

	// TODO: handling of parfor, statementblock reuse
	// TODO: collect lineage tracing and probing overhead (computation)
	
	public static void processSingleInst(Instruction inst, ExecutionContext ec, long starttime) 
	{
		// Called for every lineage tracable instruction
		if (!(inst instanceof ComputationCPInstruction))
			return;
		long computetime = System.nanoTime() - starttime;
		ComputationCPInstruction cinst = (ComputationCPInstruction) inst;
		LineageItem li = cinst.getLineageItem(ec).getValue();
		boolean isReusable = LineageCacheConfig.isReusable(inst, ec);
		// Gather the size of this intermediate
		Data data = ec.getVariable(((ComputationCPInstruction) inst).output);
		long datasize = 0;
		if (data instanceof MatrixObject)
			datasize = (((MatrixObject)data).acquireReadAndRelease()).getInMemorySize();
		else if (data instanceof ScalarObject)
			datasize = ((ScalarObject)data).getSize();
		else
			return;  // must be a frame
		probePutValue(li, computetime, datasize, isReusable);
	}

	public static void stopEstimator(List<DataIdentifier> outputs, LineageItem[]liInputs, String name) {
		// This is called for functions, before execution of the function body starts.
		// To simulate multilevel reuse, stop gathering statistics for the body
		// if the function entry is available in the cache.
		boolean allOutputsCached = true;
		for (int i=0; i<outputs.size(); i++) {
			String opcode = name + "%" + String.valueOf(i+1);
			LineageItem li = new LineageItem(opcode, liInputs);
			if (!_cache.containsKey(li))
				allOutputsCached = false;
		}
		if (allOutputsCached)
			DMLScript.LINEAGE_ESTIMATE = false;
	}

	public static void processFunc(List<DataIdentifier> outputs,
		LineageItem[] liInputs, String name, ExecutionContext ec, long computetime)
	{
		// This is called for functions, after end of execution.
		// Restart stat collection and save the function entry.
		DMLScript.LINEAGE_ESTIMATE = true;
		for (int i=0; i<outputs.size(); i++) {
			String opcode = name + "%" + String.valueOf(i+1);
			LineageItem li = new LineageItem(opcode, liInputs);
			String boundVarName = outputs.get(i).getName();
			LineageItem boundLI = ec.getLineage().get(boundVarName);
			if (boundLI != null)
				boundLI.resetVisitStatusNR();
			if (boundLI != null) {
				Data data = ec.getVariable(boundVarName);
				long datasize = 0;
				if (data instanceof MatrixObject)
					datasize = (((MatrixObject)data).acquireReadAndRelease()).getInMemorySize();
				else if (data instanceof ScalarObject)
					datasize = ((ScalarObject)data).getSize();
				else
					return;  // must be a frame
				probePutValue(li, computetime, datasize, true);
			}
		}
	}

	private static void probePutValue(LineageItem li, long computetime, long datasize, boolean isReusable) {
		// Probe the estimator cache
		if (_cache.containsKey(li)) {
			LineageEstimatorEntry ee = _cache.get(li);
			LineageEstimatorStatistics.incrementSavedComputeTime(ee.computeTime);
			// Update reused size only once per entry
			if (isReusable && ee.reuseCount == 0)
				_totReusedSize += ee.memsize;
			// Update reusecont and total time saved for this lineage item.
			ee.updateStats();
			if (_savedPerOP.containsKey(getOpcode(li))) {
				MutableTriple<String, Long, Double> op = _savedPerOP.get(getOpcode(li));
				computeSavingInst.remove(op);
				// Update saved compute time, reuse count for this operator in the queue
				op.setRight(op.getRight() + ee.computeTime*1e-6);
				op.setMiddle(op.getMiddle() + 1);
				computeSavingInst.add(op);
			}
			return;
		}

		// Put the entry in the estimator cache
		LineageEstimatorEntry ee = new LineageEstimatorEntry(li, computetime, datasize);
		_cache.put(li, ee);
		_cachesize += datasize;
		if (!_savedPerOP.containsKey(getOpcode(li))) {
			_savedPerOP.put(getOpcode(li), MutableTriple.of(getOpcode(li), 0L, 0.0));
			computeSavingInst.add(MutableTriple.of(getOpcode(li), 0L, 0.0));
		}

		if (isReusable)
			// Update cacheable size only if this entry is reusable as per 
			// the original lineage caching logic.
			_totReusableSize += ee.memsize;
		
		// Mark the count of instructions if cache is full
		if (_cacheFullCount == 0 && _cachesize >= CACHE_LIMIT)
			_cacheFullCount = _cache.size();
	}
	
	private static String getOpcode(LineageItem li) {
		String opcode = li.getOpcode();
		if (opcode.indexOf("%") == -1)
			return opcode;
		return opcode.substring(0, opcode.indexOf("%"));
	}

	public static int computeCacheFullTime() {
		double d = ((double)_cacheFullCount/_cache.size())*100;
		return (int)d;
	}
	
	public static void resetEstimatorCache() {
		_cache.clear();
		_savedPerOP.clear();
		_cachesize = 0;
		_cacheFullCount = 0;
		_totReusableSize = 0;
		_totReusedSize = 0;
	}
}
	

class LineageEstimatorEntry {
	protected LineageItem key;
	protected long computeTime;
	protected long memsize;
	protected long timestamp;
	protected long reuseCount = 0;
	protected double totTimeSaved = 0; //compute time scaled by reuse count
	
	public LineageEstimatorEntry(LineageItem li, long ct, long size) {
		key = li;
		timestamp = System.currentTimeMillis() - LineageEstimator._startTimestamp;
		computeTime = ct;
		memsize = size;
	}
	public void updateStats() {
		reuseCount++;
		totTimeSaved = computeTime*1e-6 * reuseCount; //in ms 
	}
}
