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

public class HOPStatistics {
	//HOP DAG recompile stats (potentially high update frequency)
	private static final LongAdder hopRecompileTime = new LongAdder(); //in nano sec
	private static final LongAdder hopRecompilePred = new LongAdder(); //count
	private static final LongAdder hopRecompileSB = new LongAdder();   //count


	public static void incrementHOPRecompileTime( long delta ) {
		hopRecompileTime.add(delta);
	}

	public static void incrementHOPRecompilePred() {
		hopRecompilePred.increment();
	}

	public static void incrementHOPRecompilePred(long delta) {
		hopRecompilePred.add(delta);
	}

	public static void incrementHOPRecompileSB() {
		hopRecompileSB.increment();
	}

	public static void incrementHOPRecompileSB(long delta) {
		hopRecompileSB.add(delta);
	}

	public static long getHopRecompileTime(){
		return hopRecompileTime.longValue();
	}

	public static long getHopRecompiledPredDAGs(){
		return hopRecompilePred.longValue();
	}

	public static long getHopRecompiledSBDAGs(){
		return hopRecompileSB.longValue();
	}

	public static void reset() {
		hopRecompileTime.reset();
		hopRecompilePred.reset();
		hopRecompileSB.reset();
	}

	public static String displayHOPStatistics() {
		StringBuilder sb = new StringBuilder();
		sb.append("HOP DAGs recompiled (PRED, SB):\t" + getHopRecompiledPredDAGs() + "/" + getHopRecompiledSBDAGs() + ".\n");
		sb.append("HOP DAGs recompile time:\t" + String.format("%.3f", ((double)getHopRecompileTime())/1000000000) + " sec.\n");
		return sb.toString();
	}
}
