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

public class TransformStatistics {
	private static final LongAdder transformEncoderCount = new LongAdder();

	//private static final LongAdder transformBuildTime = new LongAdder();
	private static final LongAdder transformRecodeBuildTime = new LongAdder();
	private static final LongAdder transformBinningBuildTime = new LongAdder();
	private static final LongAdder transformImputeBuildTime = new LongAdder();

	//private static final LongAdder transformApplyTime = new LongAdder();
	private static final LongAdder transformRecodeApplyTime = new LongAdder();
	private static final LongAdder transformDummyCodeApplyTime = new LongAdder();
	private static final LongAdder transformPassThroughApplyTime = new LongAdder();
	private static final LongAdder transformFeatureHashingApplyTime = new LongAdder();
	private static final LongAdder transformBinningApplyTime = new LongAdder();
	private static final LongAdder transformOmitApplyTime = new LongAdder();
	private static final LongAdder transformImputeApplyTime = new LongAdder();

	private static final LongAdder transformOutMatrixPreProcessingTime = new LongAdder();
	private static final LongAdder transformOutMatrixPostProcessingTime = new LongAdder();

	public static void incTransformEncoderCount(long encoders){
		transformEncoderCount.add(encoders);
	}

	public static void incTransformRecodeApplyTime(long t){
		transformRecodeApplyTime.add(t);
	}

	public static void incTransformDummyCodeApplyTime(long t){
		transformDummyCodeApplyTime.add(t);
	}

	public static void incTransformBinningApplyTime(long t){
		transformBinningApplyTime.add(t);
	}

	public static void incTransformPassThroughApplyTime(long t){
		transformPassThroughApplyTime.add(t);
	}

	public static void incTransformFeatureHashingApplyTime(long t){
		transformFeatureHashingApplyTime.add(t);
	}

	public static void incTransformOmitApplyTime(long t) {
		transformOmitApplyTime.add(t);
	}

	public static void incTransformImputeApplyTime(long t) {
		transformImputeApplyTime.add(t);
	}

	public static void incTransformRecodeBuildTime(long t){
		transformRecodeBuildTime.add(t);
	}

	public static void incTransformBinningBuildTime(long t){
		transformBinningBuildTime.add(t);
	}

	public static void incTransformImputeBuildTime(long t) {
		transformImputeBuildTime.add(t);
	}

	public static void incTransformOutMatrixPreProcessingTime(long t){
		transformOutMatrixPreProcessingTime.add(t);
	}

	public static void incTransformOutMatrixPostProcessingTime(long t){
		transformOutMatrixPostProcessingTime.add(t);
	}

	public static long getTransformEncodeBuildTime(){
		return transformBinningBuildTime.longValue() + transformImputeBuildTime.longValue() +
				transformRecodeBuildTime.longValue();
	}

	public static long getTransformEncodeApplyTime(){
		return transformDummyCodeApplyTime.longValue() + transformBinningApplyTime.longValue() +
				transformFeatureHashingApplyTime.longValue() + transformPassThroughApplyTime.longValue() +
				transformRecodeApplyTime.longValue() + transformOmitApplyTime.longValue() +
				transformImputeApplyTime.longValue();
	}

	public static String displayTransformStatistics() {
		if( transformEncoderCount.longValue() > 0) {
			//TODO: Cleanup and condense
			StringBuilder sb = new StringBuilder();
			sb.append("TransformEncode num. encoders:\t").append(transformEncoderCount.longValue()).append("\n");
			sb.append("TransformEncode build time:\t").append(String.format("%.3f",
				getTransformEncodeBuildTime()*1e-9)).append(" sec.\n");
			if(transformRecodeBuildTime.longValue() > 0)
				sb.append("\tRecode build time:\t").append(String.format("%.3f",
					transformRecodeBuildTime.longValue()*1e-9)).append(" sec.\n");
			if(transformBinningBuildTime.longValue() > 0)
				sb.append("\tBinning build time:\t").append(String.format("%.3f",
					transformBinningBuildTime.longValue()*1e-9)).append(" sec.\n");
			if(transformImputeBuildTime.longValue() > 0)
				sb.append("\tImpute build time:\t").append(String.format("%.3f",
					transformImputeBuildTime.longValue()*1e-9)).append(" sec.\n");

			sb.append("TransformEncode apply time:\t").append(String.format("%.3f",
				getTransformEncodeApplyTime()*1e-9)).append(" sec.\n");
			if(transformRecodeApplyTime.longValue() > 0)
				sb.append("\tRecode apply time:\t").append(String.format("%.3f",
					transformRecodeApplyTime.longValue()*1e-9)).append(" sec.\n");
			if(transformBinningApplyTime.longValue() > 0)
				sb.append("\tBinning apply time:\t").append(String.format("%.3f",
					transformBinningApplyTime.longValue()*1e-9)).append(" sec.\n");
			if(transformDummyCodeApplyTime.longValue() > 0)
				sb.append("\tDummyCode apply time:\t").append(String.format("%.3f",
					transformDummyCodeApplyTime.longValue()*1e-9)).append(" sec.\n");
			if(transformFeatureHashingApplyTime.longValue() > 0)
				sb.append("\tHashing apply time:\t").append(String.format("%.3f",
					transformFeatureHashingApplyTime.longValue()*1e-9)).append(" sec.\n");
			if(transformPassThroughApplyTime.longValue() > 0)
				sb.append("\tPassThrough apply time:\t").append(String.format("%.3f",
					transformPassThroughApplyTime.longValue()*1e-9)).append(" sec.\n");
			if(transformOmitApplyTime.longValue() > 0)
				sb.append("\tOmit apply time:\t").append(String.format("%.3f",
					transformOmitApplyTime.longValue()*1e-9)).append(" sec.\n");
			if(transformImputeApplyTime.longValue() > 0)
				sb.append("\tImpute apply time:\t").append(String.format("%.3f",
					transformImputeApplyTime.longValue()*1e-9)).append(" sec.\n");

			sb.append("TransformEncode PreProc. time:\t").append(String.format("%.3f",
				transformOutMatrixPreProcessingTime.longValue()*1e-9)).append(" sec.\n");
			sb.append("TransformEncode PostProc. time:\t").append(String.format("%.3f",
				transformOutMatrixPostProcessingTime.longValue()*1e-9)).append(" sec.\n");
			return sb.toString();
		}
		return "";
	}
}
