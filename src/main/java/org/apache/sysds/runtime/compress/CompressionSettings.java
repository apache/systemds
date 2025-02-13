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

package org.apache.sysds.runtime.compress;

import java.util.EnumSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.cocode.CoCoderFactory.PartitionerType;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.insertionsort.InsertionSorterFactory.SORT_TYPE;
import org.apache.sysds.runtime.compress.cost.CostEstimatorFactory.CostType;
import org.apache.sysds.runtime.compress.estim.sample.SampleEstimatorFactory.EstimationType;

/**
 * Compression Settings class, used as a bundle of parameters inside the Compression framework. See
 * CompressionSettingsBuilder for default non static parameters.
 */
public class CompressionSettings {
	private static final Log LOG = LogFactory.getLog(CompressionSettings.class.getName());

	/** Parallelization threshold for DDC compression */
	public static int PAR_DDC_THRESHOLD = 10000;

	/**
	 * Size of the blocks used in a blocked bitmap representation. Note it is exactly Character.MAX_VALUE. This is not
	 * Character max value + 1 because it breaks the offsets in cases with fully dense values.
	 */
	public static final int BITMAP_BLOCK_SZ = Character.MAX_VALUE;

	/**
	 * Sorting of values by physical length helps by 10-20%, especially for serial, while slight performance decrease
	 * for parallel incl multi-threaded, hence not applied for distributed operations (also because compression time +
	 * garbage collection increases)
	 */
	public final boolean sortTuplesByFrequency;

	/**
	 * The sampling ratio used when choosing ColGroups. Note that, default behavior is to use exact estimator if the
	 * number of elements is below 1000.
	 * 
	 * DEPRECATED
	 */
	public final double samplingRatio;

	/**
	 * The sampling ratio power to use when choosing sample size. This is used in accordance to the function:
	 * 
	 * sampleSize += nRows^samplePower;
	 * 
	 * The value is bounded to be in the range of 0 to 1, 1 giving a sample size of everything, and 0 adding 1.
	 */
	public final double samplePower;

	/** Share DDC Dictionaries between ColGroups. */
	public final boolean allowSharedDictionary;

	/** Boolean specifying which transpose setting is used, can be auto, true or false */
	public final String transposeInput;

	/** If the seed is -1 then the system used system millisecond time and class hash for seeding. */
	public final int seed;

	/** True if lossy compression is enabled */
	public final boolean lossy;

	/** The selected method for column partitioning used in CoCoding compressed columns */
	public final PartitionerType columnPartitioner;

	/** The cost computation type for the compression */
	public final CostType costComputationType;

	/** The maximum number of columns CoCoded allowed */
	public final int maxColGroupCoCode;

	/**
	 * A Cocode parameter that differ in behavior based on compression method, in general it is a value that reflects
	 * aggressively likely coCoding is used.
	 */
	public final double coCodePercentage;

	/**
	 * Valid Compressions List, containing the ColGroup CompressionTypes that are allowed to be used for the compression
	 * Default is to always allow for Uncompromisable ColGroup.
	 */
	public final EnumSet<CompressionType> validCompressions;

	/** The minimum size of the sample extracted. */
	public final int minimumSampleSize;

	/** The maximum size of the sample extracted. */
	public final int maxSampleSize;

	/** The sample type used for sampling */
	public final EstimationType estimationType;

	/**
	 * Transpose input matrix, to optimize access when extracting bitmaps. This setting is changed inside the script
	 * based on the transposeInput setting.
	 * 
	 * This is intentionally left as a mutable value, since the transposition of the input matrix is decided in phase 3.
	 */
	public boolean transposed = false;

	/** The minimum compression ratio to achieve. */
	public final double minimumCompressionRatio;

	/** Is a spark instruction */
	public final boolean isInSparkInstruction;

	/** The sorting type used in sorting/joining offsets to create SDC groups */
	public final SORT_TYPE sdcSortType;

	/** if the settings have been logged already. */
	public static boolean printedStatus = false;

	public final double[] scaleFactors;

	protected CompressionSettings(double samplingRatio, double samplePower, boolean allowSharedDictionary,
		String transposeInput, int seed, boolean lossy, EnumSet<CompressionType> validCompressions,
		boolean sortValuesByLength, PartitionerType columnPartitioner, int maxColGroupCoCode, double coCodePercentage,
		int minimumSampleSize, int maxSampleSize, EstimationType estimationType, CostType costComputationType,
		double minimumCompressionRatio, boolean isInSparkInstruction, SORT_TYPE sdcSortType, double[] scaleFactors) {
		this.samplingRatio = samplingRatio;
		this.samplePower = samplePower;
		this.allowSharedDictionary = allowSharedDictionary;
		this.transposeInput = transposeInput;
		this.seed = seed == -1 ? (int) System.nanoTime() : seed;
		this.validCompressions = validCompressions;
		this.lossy = lossy;
		this.sortTuplesByFrequency = sortValuesByLength;
		this.columnPartitioner = columnPartitioner;
		this.maxColGroupCoCode = maxColGroupCoCode;
		this.coCodePercentage = coCodePercentage;
		this.minimumSampleSize = minimumSampleSize;
		this.maxSampleSize = maxSampleSize;
		this.estimationType = estimationType;
		this.costComputationType = costComputationType;
		this.minimumCompressionRatio = minimumCompressionRatio;
		this.isInSparkInstruction = isInSparkInstruction;
		this.sdcSortType = sdcSortType;
		this.scaleFactors = scaleFactors;
		
		if(!printedStatus && LOG.isDebugEnabled()) {
			printedStatus = true;
			LOG.debug(this.toString());
		}
	}

	public boolean isRLEAllowed() {
		return this.validCompressions.contains(CompressionType.RLE);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("CompressionSettings: ");
		sb.append("\t Valid Compressions: " + validCompressions);
		sb.append("\t Share dict: " + allowSharedDictionary);
		sb.append("\t Partitioner: " + columnPartitioner);
		sb.append("\t Lossy: " + lossy);
		sb.append("\t Cost Computation Type: " + costComputationType);
		if(samplingRatio < 1.0)
			sb.append("\t Estimation Type: " + estimationType);
		return sb.toString();
	}
}
