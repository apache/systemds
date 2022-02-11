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
	private static final LongAdder compileTime = new LongAdder(); //in nano
	private static final LongAdder classCompileTime = new LongAdder(); //in nano
	private static final LongAdder hopCompile = new LongAdder(); //count
	private static final LongAdder cPlanCompile = new LongAdder(); //count
	private static final LongAdder classCompile = new LongAdder(); //count
	private static final LongAdder enumAll = new LongAdder(); //count
	private static final LongAdder enumAllP = new LongAdder(); //count
	private static final LongAdder enumEval = new LongAdder(); //count
	private static final LongAdder enumEvalP = new LongAdder(); //count
	private static final LongAdder opCacheHits = new LongAdder(); //count
	private static final LongAdder opCacheTotal = new LongAdder(); //count
	private static final LongAdder planCacheHits = new LongAdder(); //count
	private static final LongAdder planCacheTotal = new LongAdder(); //count


	public static void incrementDAGCompile() {
		hopCompile.increment();
	}

	public static void incrementCPlanCompile(long delta) {
		cPlanCompile.add(delta);
	}

	public static void incrementEnumAll(long delta) {
		enumAll.add(delta);
	}
	public static void incrementEnumAllP(long delta) {
		enumAllP.add(delta);
	}
	public static void incrementEnumEval(long delta) {
		enumEval.add(delta);
	}
	public static void incrementEnumEvalP(long delta) {
		enumEvalP.add(delta);
	}

	public static void incrementClassCompile() {
		classCompile.increment();
	}

	public static void incrementCompileTime(long delta) {
		compileTime.add(delta);
	}

	public static void incrementClassCompileTime(long delta) {
		classCompileTime.add(delta);
	}

	public static void incrementOpCacheHits() {
		opCacheHits.increment();
	}

	public static void incrementOpCacheTotal() {
		opCacheTotal.increment();
	}

	public static void incrementPlanCacheHits() {
		planCacheHits.increment();
	}

	public static void incrementPlanCacheTotal() {
		planCacheTotal.increment();
	}

	public static long getDAGCompile() {
		return hopCompile.longValue();
	}

	public static long getCPlanCompile() {
		return cPlanCompile.longValue();
	}

	public static long getEnumAll() {
		return enumAll.longValue();
	}

	public static long getEnumAllP() {
		return enumAllP.longValue();
	}

	public static long getEnumEval() {
		return enumEval.longValue();
	}

	public static long getEnumEvalP() {
		return enumEvalP.longValue();
	}

	public static long getClassCompile() {
		return classCompile.longValue();
	}

	public static long getCompileTime() {
		return compileTime.longValue();
	}

	public static long getClassCompileTime() {
		return classCompileTime.longValue();
	}

	public static long getOpCacheHits() {
		return opCacheHits.longValue();
	}

	public static long getOpCacheTotal() {
		return opCacheTotal.longValue();
	}

	public static long getPlanCacheHits() {
		return planCacheHits.longValue();
	}

	public static long getPlanCacheTotal() {
		return planCacheTotal.longValue();
	}

	public static void reset() {
		hopCompile.reset();
		cPlanCompile.reset();
		classCompile.reset();
		enumAll.reset();
		enumAllP.reset();
		enumEval.reset();
		enumEvalP.reset();
		compileTime.reset();
		classCompileTime.reset();
		opCacheHits.reset();
		opCacheTotal.reset();
		planCacheHits.reset();
		planCacheTotal.reset();
	}

	public static String displayStatistics() {
		StringBuilder sb = new StringBuilder();
		sb.append("Codegen compile (DAG,CP,JC):\t" + getDAGCompile() + "/"
				+ getCPlanCompile() + "/" + getClassCompile() + ".\n");
		sb.append("Codegen enum (ALLt/p,EVALt/p):\t" + getEnumAll() + "/" +
				getEnumAllP() + "/" + getEnumEval() + "/" + getEnumEvalP() + ".\n");
		sb.append("Codegen compile times (DAG,JC):\t" + String.format("%.3f", (double)getCompileTime()/1000000000) + "/" +
				String.format("%.3f", (double)getClassCompileTime()/1000000000)  + " sec.\n");
		sb.append("Codegen enum plan cache hits:\t" + getPlanCacheHits() + "/" + getPlanCacheTotal() + ".\n");
		sb.append("Codegen op plan cache hits:\t" + getOpCacheHits() + "/" + getOpCacheTotal() + ".\n");
		return sb.toString();
	}
}
