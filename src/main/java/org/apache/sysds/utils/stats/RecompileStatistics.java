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

public class RecompileStatistics {
	//HOP DAG recompile stats (potentially high update frequency)
	private static final LongAdder recompileTime = new LongAdder(); //in nano sec
	private static final LongAdder recompilePred = new LongAdder(); //count
	private static final LongAdder recompileSB = new LongAdder();   //count


	public static void incrementRecompileTime( long delta ) {
		recompileTime.add(delta);
	}

	public static void incrementRecompilePred() {
		recompilePred.increment();
	}

	public static void incrementRecompilePred(long delta) {
		recompilePred.add(delta);
	}

	public static void incrementRecompileSB() {
		recompileSB.increment();
	}

	public static void incrementRecompileSB(long delta) {
		recompileSB.add(delta);
	}

	public static long getRecompileTime(){
		return recompileTime.longValue();
	}

	public static long getRecompiledPredDAGs(){
		return recompilePred.longValue();
	}

	public static long getRecompiledSBDAGs(){
		return recompileSB.longValue();
	}

	public static void reset() {
		recompileTime.reset();
		recompilePred.reset();
		recompileSB.reset();
	}

	public static String displayStatistics() {
		StringBuilder sb = new StringBuilder();
		sb.append("HOP DAGs recompiled (PRED, SB):\t" + getRecompiledPredDAGs() + "/" + getRecompiledSBDAGs() + ".\n");
		sb.append("HOP DAGs recompile time:\t" + String.format("%.3f", ((double)getRecompileTime())/1000000000) + " sec.\n");
		return sb.toString();
	}
}
