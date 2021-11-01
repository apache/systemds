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

package org.apache.sysds.utils.stats;

import java.util.concurrent.atomic.LongAdder;

public class CodegenStatistics {
	private static final LongAdder codegenCompileTime = new LongAdder(); //in nano
	private static final LongAdder codegenClassCompileTime = new LongAdder(); //in nano
	private static final LongAdder codegenHopCompile = new LongAdder(); //count
	private static final LongAdder codegenCPlanCompile = new LongAdder(); //count
	private static final LongAdder codegenClassCompile = new LongAdder(); //count
	private static final LongAdder codegenEnumAll = new LongAdder(); //count
	private static final LongAdder codegenEnumAllP = new LongAdder(); //count
	private static final LongAdder codegenEnumEval = new LongAdder(); //count
	private static final LongAdder codegenEnumEvalP = new LongAdder(); //count
	private static final LongAdder codegenOpCacheHits = new LongAdder(); //count
	private static final LongAdder codegenOpCacheTotal = new LongAdder(); //count
	private static final LongAdder codegenPlanCacheHits = new LongAdder(); //count
	private static final LongAdder codegenPlanCacheTotal = new LongAdder(); //count


	public static void incrementCodegenDAGCompile() {
		codegenHopCompile.increment();
	}

	public static void incrementCodegenCPlanCompile(long delta) {
		codegenCPlanCompile.add(delta);
	}

	public static void incrementCodegenEnumAll(long delta) {
		codegenEnumAll.add(delta);
	}
	public static void incrementCodegenEnumAllP(long delta) {
		codegenEnumAllP.add(delta);
	}
	public static void incrementCodegenEnumEval(long delta) {
		codegenEnumEval.add(delta);
	}
	public static void incrementCodegenEnumEvalP(long delta) {
		codegenEnumEvalP.add(delta);
	}

	public static void incrementCodegenClassCompile() {
		codegenClassCompile.increment();
	}

	public static void incrementCodegenCompileTime(long delta) {
		codegenCompileTime.add(delta);
	}

	public static void incrementCodegenClassCompileTime(long delta) {
		codegenClassCompileTime.add(delta);
	}

	public static void incrementCodegenOpCacheHits() {
		codegenOpCacheHits.increment();
	}

	public static void incrementCodegenOpCacheTotal() {
		codegenOpCacheTotal.increment();
	}

	public static void incrementCodegenPlanCacheHits() {
		codegenPlanCacheHits.increment();
	}

	public static void incrementCodegenPlanCacheTotal() {
		codegenPlanCacheTotal.increment();
	}

	public static long getCodegenDAGCompile() {
		return codegenHopCompile.longValue();
	}

	public static long getCodegenCPlanCompile() {
		return codegenCPlanCompile.longValue();
	}

	public static long getCodegenEnumAll() {
		return codegenEnumAll.longValue();
	}

	public static long getCodegenEnumAllP() {
		return codegenEnumAllP.longValue();
	}

	public static long getCodegenEnumEval() {
		return codegenEnumEval.longValue();
	}

	public static long getCodegenEnumEvalP() {
		return codegenEnumEvalP.longValue();
	}

	public static long getCodegenClassCompile() {
		return codegenClassCompile.longValue();
	}

	public static long getCodegenCompileTime() {
		return codegenCompileTime.longValue();
	}

	public static long getCodegenClassCompileTime() {
		return codegenClassCompileTime.longValue();
	}

	public static long getCodegenOpCacheHits() {
		return codegenOpCacheHits.longValue();
	}

	public static long getCodegenOpCacheTotal() {
		return codegenOpCacheTotal.longValue();
	}

	public static long getCodegenPlanCacheHits() {
		return codegenPlanCacheHits.longValue();
	}

	public static long getCodegenPlanCacheTotal() {
		return codegenPlanCacheTotal.longValue();
	}

	public static void reset() {
		codegenHopCompile.reset();
		codegenCPlanCompile.reset();
		codegenClassCompile.reset();
		codegenEnumAll.reset();
		codegenEnumAllP.reset();
		codegenEnumEval.reset();
		codegenEnumEvalP.reset();
		codegenCompileTime.reset();
		codegenClassCompileTime.reset();
		codegenOpCacheHits.reset();
		codegenOpCacheTotal.reset();
		codegenPlanCacheHits.reset();
		codegenPlanCacheTotal.reset();
	}

	public static String displayCodegenStatistics() {
		StringBuilder sb = new StringBuilder();
		sb.append("Codegen compile (DAG,CP,JC):\t" + getCodegenDAGCompile() + "/"
				+ getCodegenCPlanCompile() + "/" + getCodegenClassCompile() + ".\n");
		sb.append("Codegen enum (ALLt/p,EVALt/p):\t" + getCodegenEnumAll() + "/" +
				getCodegenEnumAllP() + "/" + getCodegenEnumEval() + "/" + getCodegenEnumEvalP() + ".\n");
		sb.append("Codegen compile times (DAG,JC):\t" + String.format("%.3f", (double)getCodegenCompileTime()/1000000000) + "/" +
				String.format("%.3f", (double)getCodegenClassCompileTime()/1000000000)  + " sec.\n");
		sb.append("Codegen enum plan cache hits:\t" + getCodegenPlanCacheHits() + "/" + getCodegenPlanCacheTotal() + ".\n");
		sb.append("Codegen op plan cache hits:\t" + getCodegenOpCacheHits() + "/" + getCodegenOpCacheTotal() + ".\n");
		return sb.toString();
	}
}
