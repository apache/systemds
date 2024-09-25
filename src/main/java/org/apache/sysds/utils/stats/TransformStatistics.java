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
	private static final LongAdder encoderCount = new LongAdder();

	//private static final LongAdder buildTime = new LongAdder();
	private static final LongAdder recodeBuildTime = new LongAdder();
	private static final LongAdder binningBuildTime = new LongAdder();
	private static final LongAdder bowBuildTime = new LongAdder();
	private static final LongAdder imputeBuildTime = new LongAdder();

	//private static final LongAdder applyTime = new LongAdder();
	private static final LongAdder recodeApplyTime = new LongAdder();
	private static final LongAdder dummyCodeApplyTime = new LongAdder();

	private static final LongAdder wordEmbeddingApplyTime = new LongAdder();
	private static final LongAdder bagOfWordsApplyTime = new LongAdder();
	private static final LongAdder passThroughApplyTime = new LongAdder();
	private static final LongAdder featureHashingApplyTime = new LongAdder();
	private static final LongAdder binningApplyTime = new LongAdder();
	private static final LongAdder UDFApplyTime = new LongAdder();
	private static final LongAdder omitApplyTime = new LongAdder();
	private static final LongAdder imputeApplyTime = new LongAdder();

	private static final LongAdder outMatrixPreProcessingTime = new LongAdder();
	private static final LongAdder outMatrixPostProcessingTime = new LongAdder();
	private static final LongAdder mapSizeEstimationTime = new LongAdder();

	public static void incEncoderCount(long encoders) {
		encoderCount.add(encoders);
	}

	public static void incRecodeApplyTime(long t) {
		recodeApplyTime.add(t);
	}

	public static void incDummyCodeApplyTime(long t) {
		dummyCodeApplyTime.add(t);
	}

	public static void incWordEmbeddingApplyTime(long t){
		wordEmbeddingApplyTime.add(t);
	}

	public static void incBagOfWordsApplyTime(long t){
		bagOfWordsApplyTime.add(t);
	}


	public static void incBinningApplyTime(long t) {
		binningApplyTime.add(t);
	}

	public static void incUDFApplyTime(long t) {
		UDFApplyTime.add(t);
	}

	public static void incPassThroughApplyTime(long t) {
		passThroughApplyTime.add(t);
	}

	public static void incFeatureHashingApplyTime(long t) {
		featureHashingApplyTime.add(t);
	}

	public static void incOmitApplyTime(long t) {
		omitApplyTime.add(t);
	}

	public static void incImputeApplyTime(long t) {
		imputeApplyTime.add(t);
	}

	public static void incRecodeBuildTime(long t) {
		recodeBuildTime.add(t);
	}

	public static void incBinningBuildTime(long t) {
		binningBuildTime.add(t);
	}

	public static void incBagOfWordsBuildTime(long t) {
		bowBuildTime.add(t);
	}

	public static void incImputeBuildTime(long t) {
		imputeBuildTime.add(t);
	}

	public static void incOutMatrixPreProcessingTime(long t) {
		outMatrixPreProcessingTime.add(t);
	}

	public static void incOutMatrixPostProcessingTime(long t) {
		outMatrixPostProcessingTime.add(t);
	}

	public static void incMapSizeEstimationTime(long t) {
		mapSizeEstimationTime.add(t);
	}

	public static long getEncodeBuildTime() {
		return binningBuildTime.longValue() + imputeBuildTime.longValue() +
				recodeBuildTime.longValue() + bowBuildTime.longValue();
	}

	public static long getEncodeApplyTime() {
		return dummyCodeApplyTime.longValue() + binningApplyTime.longValue() +
				featureHashingApplyTime.longValue() + passThroughApplyTime.longValue() +
				recodeApplyTime.longValue() + UDFApplyTime.longValue() +
				omitApplyTime.longValue() + imputeApplyTime.longValue() + wordEmbeddingApplyTime.longValue() +
				bagOfWordsApplyTime.longValue();
	}

	public static void reset() {
		encoderCount.reset();
		// buildTime.reset();
		recodeBuildTime.reset();
		binningBuildTime.reset();
		imputeBuildTime.reset();
		bowBuildTime.reset();
		// applyTime.reset();
		recodeApplyTime.reset();
		dummyCodeApplyTime.reset();
		passThroughApplyTime.reset();
		featureHashingApplyTime.reset();
		bagOfWordsApplyTime.reset();
		wordEmbeddingApplyTime.reset();
		binningApplyTime.reset();
		UDFApplyTime.reset();
		omitApplyTime.reset();
		imputeApplyTime.reset();
		outMatrixPreProcessingTime.reset();
		outMatrixPostProcessingTime.reset();
		mapSizeEstimationTime.reset();
	}

	public static String displayStatistics() {
		if( encoderCount.longValue() > 0) {
			//TODO: Cleanup and condense
			StringBuilder sb = new StringBuilder();
			sb.append("TransformEncode num. encoders:\t").append(encoderCount.longValue()).append("\n");
			sb.append("TransformEncode build time:\t").append(String.format("%.3f",
				getEncodeBuildTime()*1e-9)).append(" sec.\n");
			if(recodeBuildTime.longValue() > 0)
				sb.append("\tRecode build time:\t").append(String.format("%.3f",
					recodeBuildTime.longValue()*1e-9)).append(" sec.\n");
			if(binningBuildTime.longValue() > 0)
				sb.append("\tBinning build time:\t").append(String.format("%.3f",
					binningBuildTime.longValue()*1e-9)).append(" sec.\n");
			if(imputeBuildTime.longValue() > 0)
				sb.append("\tImpute build time:\t").append(String.format("%.3f",
					imputeBuildTime.longValue()*1e-9)).append(" sec.\n");
			if(bowBuildTime.longValue() > 0)
				sb.append("\tBagOfWords build time:\t").append(String.format("%.3f",
						bowBuildTime.longValue()*1e-9)).append(" sec.\n");

			sb.append("TransformEncode apply time:\t").append(String.format("%.3f",
				getEncodeApplyTime()*1e-9)).append(" sec.\n");
			if(recodeApplyTime.longValue() > 0)
				sb.append("\tRecode apply time:\t").append(String.format("%.3f",
					recodeApplyTime.longValue()*1e-9)).append(" sec.\n");
			if(binningApplyTime.longValue() > 0)
				sb.append("\tBinning apply time:\t").append(String.format("%.3f",
					binningApplyTime.longValue()*1e-9)).append(" sec.\n");
			if(dummyCodeApplyTime.longValue() > 0)
				sb.append("\tDummyCode apply time:\t").append(String.format("%.3f",
					dummyCodeApplyTime.longValue()*1e-9)).append(" sec.\n");
			if(wordEmbeddingApplyTime.longValue() > 0)
				sb.append("\tWordEmbedding apply time:\t").append(String.format("%.3f",
						wordEmbeddingApplyTime.longValue()*1e-9)).append(" sec.\n");
			if(bagOfWordsApplyTime.longValue() > 0)
				sb.append("\tBagOfWords apply time:\t").append(String.format("%.3f",
						bagOfWordsApplyTime.longValue()*1e-9)).append(" sec.\n");
			if(featureHashingApplyTime.longValue() > 0)
				sb.append("\tHashing apply time:\t").append(String.format("%.3f",
					featureHashingApplyTime.longValue()*1e-9)).append(" sec.\n");
			if(passThroughApplyTime.longValue() > 0)
				sb.append("\tPassThrough apply time:\t").append(String.format("%.3f",
					passThroughApplyTime.longValue()*1e-9)).append(" sec.\n");
			if(UDFApplyTime.longValue() > 0)
				sb.append("\tUDF apply time:\t").append(String.format("%.3f",
					UDFApplyTime.longValue()*1e-9)).append(" sec.\n");
			if(omitApplyTime.longValue() > 0)
				sb.append("\tOmit apply time:\t").append(String.format("%.3f",
					omitApplyTime.longValue()*1e-9)).append(" sec.\n");
			if(imputeApplyTime.longValue() > 0)
				sb.append("\tImpute apply time:\t").append(String.format("%.3f",
					imputeApplyTime.longValue()*1e-9)).append(" sec.\n");

			sb.append("TransformEncode PreProc. time:\t").append(String.format("%.3f",
				outMatrixPreProcessingTime.longValue()*1e-9)).append(" sec.\n");
			sb.append("TransformEncode PostProc. time:\t").append(String.format("%.3f",
				outMatrixPostProcessingTime.longValue()*1e-9)).append(" sec.\n");
			if(mapSizeEstimationTime.longValue() > 0)
				sb.append("TransformEncode SizeEst. time:\t").append(String.format("%.3f",
					mapSizeEstimationTime.longValue()*1e-9)).append(" sec.\n");
			return sb.toString();
		}
		return "";
	}
}
