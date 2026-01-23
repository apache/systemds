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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.compress.cocode.CoCoderFactory.PartitionerType;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.insertionsort.InsertionSorterFactory.SORT_TYPE;
import org.apache.sysds.runtime.compress.cost.CostEstimatorFactory.CostType;
import org.apache.sysds.runtime.compress.estim.sample.SampleEstimatorFactory.EstimationType;

/**
 * Builder pattern for Compression Settings. See CompressionSettings for details on values.
 */
public class CompressionSettingsBuilder {
	private double samplingRatio;
	private double samplePower = 0.65;
	private boolean allowSharedDictionary = false;
	private String transposeInput;
	private int seed = -1;
	private boolean lossy = false;
	private EnumSet<CompressionType> validCompressions;
	private boolean sortValuesByLength = true;
	private int maxColGroupCoCode = 10000;
	private double coCodePercentage = 0.01;
	private int minimumSampleSize = 3000;
	private int maxSampleSize = 1000000;
	private EstimationType estimationType = EstimationType.HassAndStokes;
	private PartitionerType columnPartitioner;
	private CostType costType;
	private double minimumCompressionRatio = 1.0;
	private boolean isInSparkInstruction = false;
	private SORT_TYPE sdcSortType = SORT_TYPE.MATERIALIZE;
	private double[] scaleFactors = null;
	private boolean preferDeltaEncoding = false;

    public CompressionSettingsBuilder() {

		DMLConfig conf = ConfigurationManager.getDMLConfig();
		this.lossy = conf.getBooleanValue(DMLConfig.COMPRESSED_LOSSY);
		this.validCompressions = EnumSet.of(CompressionType.UNCOMPRESSED, CompressionType.CONST, CompressionType.EMPTY);
		String[] validCompressionsString = conf.getTextValue(DMLConfig.COMPRESSED_VALID_COMPRESSIONS).split(",");
		for(String comp : validCompressionsString)
			validCompressions.add(CompressionType.valueOf(comp));
		samplingRatio = conf.getDoubleValue(DMLConfig.COMPRESSED_SAMPLING_RATIO);
		columnPartitioner = PartitionerType.valueOf(conf.getTextValue(DMLConfig.COMPRESSED_COCODE));
		costType = CostType.valueOf(conf.getTextValue(DMLConfig.COMPRESSED_COST_MODEL));
		transposeInput = conf.getTextValue(DMLConfig.COMPRESSED_TRANSPOSE);
		seed = DMLScript.SEED;

	}

	/**
	 * Sets the scale factors for compression, enabling quantization-fused compression.
	 * 
	 * @param scaleFactors An array of scale factors applied during compression. 
	 *                     - If row-wise scaling is used, this should be an array where each value corresponds to a row. 
	 *                     - If a single scalar is provided, it is applied uniformly to the entire matrix.
	 * @return The CompressionSettingsBuilder instance with the updated scale factors.
	 */
	public CompressionSettingsBuilder setScaleFactor(double[] scaleFactors) {
		this.scaleFactors = scaleFactors;
		return this;
	}

	/**
	 * Copy the settings from another CompressionSettings Builder, modifies this, not that.
	 * 
	 * @param that The other CompressionSettingsBuilder to copy settings from.
	 * @return The modified CompressionSettings in the same object.
	 */
	public CompressionSettingsBuilder copySettings(CompressionSettings that) {
		this.samplingRatio = that.samplingRatio;
		this.allowSharedDictionary = that.allowSharedDictionary;
		this.transposeInput = that.transposeInput;
		this.seed = that.seed;
		this.lossy = that.lossy;
		this.validCompressions = EnumSet.copyOf(that.validCompressions);
		this.sortValuesByLength = that.sortTuplesByFrequency;
		this.columnPartitioner = that.columnPartitioner;
		this.maxColGroupCoCode = that.maxColGroupCoCode;
		this.coCodePercentage = that.coCodePercentage;
		this.minimumSampleSize = that.minimumSampleSize;
		this.preferDeltaEncoding = that.preferDeltaEncoding;
		return this;
	}

	/**
	 * Set the Compression to use Lossy compression.
	 * 
	 * @param lossy A boolean specifying if the compression should be lossy
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setLossy(boolean lossy) {
		this.lossy = lossy;
		return this;
	}

	/**
	 * Set the sampling ratio in percent to sample the input matrix. Input value should be in range 0.0 - 1.0
	 * 
	 * @param samplingRatio The ratio to sample from the input
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setSamplingRatio(double samplingRatio) {
		this.samplingRatio = samplingRatio;
		return this;
	}

	/**
	 * Set the sortValuesByLength flag. This sorts the dictionaries containing the data based on their occurences in the
	 * ColGroup. Improving cache efficiency especially for diverse column groups.
	 * 
	 * @param sortValuesByLength A boolean specifying if the values should be sorted
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setSortValuesByLength(boolean sortValuesByLength) {
		this.sortValuesByLength = sortValuesByLength;
		return this;
	}

	/**
	 * Allow the Dictionaries to be shared between different column groups.
	 * 
	 * @param allowSharedDictionary A boolean specifying if the dictionary can be shared between column groups.
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setAllowSharedDictionary(boolean allowSharedDictionary) {
		this.allowSharedDictionary = allowSharedDictionary;
		return this;
	}

	/**
	 * Specify if the input matrix should be transposed before compression. This improves cache efficiency while
	 * compression the input matrix
	 * 
	 * @param transposeInput string specifying if the input should be transposed before compression, should be one of
	 *                       "auto", "true" or "false"
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setTransposeInput(String transposeInput) {
		switch(transposeInput) {
			case "auto":
			case "true":
			case "false":
				this.transposeInput = transposeInput;
				break;
			default:
				throw new DMLCompressionException("Invalid transpose technique");
		}
		return this;
	}

	/**
	 * Set the seed for the compression operation.
	 * 
	 * @param seed The seed used in sampling the matrix and general operations in the compression.
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setSeed(int seed) {
		this.seed = seed;
		return this;
	}

	/**
	 * Set the valid compression strategies used for the compression.
	 * 
	 * @param validCompressions An EnumSet of CompressionTypes to use in the compression
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setValidCompressions(EnumSet<CompressionType> validCompressions) {
		// should always contain Uncompressed as an option.
		if(!validCompressions.contains(CompressionType.UNCOMPRESSED))
			validCompressions.add(CompressionType.UNCOMPRESSED);
		if(!validCompressions.contains(CompressionType.CONST))
			validCompressions.add(CompressionType.CONST);
		if(!validCompressions.contains(CompressionType.EMPTY))
			validCompressions.add(CompressionType.EMPTY);
		this.validCompressions = validCompressions;
		return this;
	}

	/**
	 * Add a single valid compression type to the EnumSet of valid compressions.
	 * 
	 * @param cp The compression type to add to the valid ones.
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder addValidCompression(CompressionType cp) {
		this.validCompressions.add(cp);
		return this;
	}

    /**
     * Ziel-Gesamtverlust für piecewise lineare Kompression.
     * Interpretation: maximal erlaubter globaler MSE pro Wert in der Spalte.
     * 0.0  ~ quasi verlustfrei, viele Segmente
     * >0   ~ mehr Approximation erlaubt, weniger Segmente


    public void setPiecewiseTargetLoss(double piecewiseTargetLoss) {
        this.piecewiseTargetLoss = piecewiseTargetLoss;
    }*/

	/**
	 * Clear all the compression types allowed in the compression. This will only allow the Uncompressed ColGroup type.
	 * Since this is required for operation of the compression
	 * 
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder clearValidCompression() {
		this.validCompressions = EnumSet.of(CompressionType.UNCOMPRESSED, CompressionType.EMPTY, CompressionType.CONST);
		return this;
	}

	/**
	 * Set the type of CoCoding Partitioner type to use for combining columns together.
	 * 
	 * @param columnPartitioner The Strategy to select from PartitionerType
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setColumnPartitioner(PartitionerType columnPartitioner) {
		this.columnPartitioner = columnPartitioner;
		return this;
	}

	/**
	 * Set the maximum number of columns to CoCode together in the CoCoding strategy. Compression time increase with
	 * higher numbers.
	 * 
	 * @param maxColGroupCoCode The max selected.
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setMaxColGroupCoCode(int maxColGroupCoCode) {
		this.maxColGroupCoCode = maxColGroupCoCode;
		return this;
	}

	/**
	 * Set the coCode percentage, the effect is different based on the coCoding strategy, but the general effect is that
	 * higher values results in more coCoding while lower values result in less.
	 * 
	 * Note that with high coCoding the compression ratio would possibly be lower.
	 * 
	 * @param coCodePercentage The percentage to set.
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setCoCodePercentage(double coCodePercentage) {
		this.coCodePercentage = coCodePercentage;
		return this;
	}

	/**
	 * Set the minimum sample size to extract from a given matrix, this overrules the sample percentage if the sample
	 * percentage extracted is lower than this minimum bound.
	 * 
	 * @param minimumSampleSize The minimum sample size to extract
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setMinimumSampleSize(int minimumSampleSize) {
		this.minimumSampleSize = minimumSampleSize;
		return this;
	}

	/**
	 * Set the maximum sample size to extract from a given matrix, this overrules the sample percentage if the sample
	 * percentage extracted is higher than this maximum bound.
	 * 
	 * @param maxSampleSize The maximum sample size to extract
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setMaxSampleSize(int maxSampleSize) {
		this.maxSampleSize = maxSampleSize;
		return this;
	}

	/**
	 * Set the estimation type used for the sampled estimates.
	 * 
	 * @param estimationType the estimation type in used.
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setEstimationType(EstimationType estimationType) {
		this.estimationType = estimationType;
		return this;
	}

	/**
	 * Set the cost type used for estimating the cost of column groups default is memory based.
	 * 
	 * @param costType The Cost type wanted
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setCostType(CostType costType) {
		this.costType = costType;
		return this;
	}

	/**
	 * Set the minimum compression ratio to be achieved by the compression.
	 * 
	 * @param ratio The ratio to achieve while compressing
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setMinimumCompressionRatio(double ratio) {
		this.minimumCompressionRatio = ratio;
		return this;
	}

	/**
	 * Inform the compression that it is executed in a spark instruction.
	 * 
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setIsInSparkInstruction() {
		this.isInSparkInstruction = true;
		return this;
	}

	/**
	 * Set the sort type to use.
	 * 
	 * @param sdcSortType The sort type for the construction of SDC groups
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setSDCSortType(SORT_TYPE sdcSortType) {
		this.sdcSortType = sdcSortType;
		return this;
	}

	/**
	 * Set whether to prefer delta encoding during compression estimation.
	 * When enabled, the compression estimator will use delta encoding statistics
	 * instead of regular encoding statistics.
	 * 
	 * @param preferDeltaEncoding Whether to prefer delta encoding
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setPreferDeltaEncoding(boolean preferDeltaEncoding) {
		this.preferDeltaEncoding = preferDeltaEncoding;
		return this;
	}

	/**
	 * Create the CompressionSettings object to use in the compression.
	 * 
	 * @return The CompressionSettings
	 */
	public CompressionSettings create() {
		return new CompressionSettings(samplingRatio, samplePower, allowSharedDictionary, transposeInput, seed, lossy,
			validCompressions, sortValuesByLength, columnPartitioner, maxColGroupCoCode, coCodePercentage,
			minimumSampleSize, maxSampleSize, estimationType, costType, minimumCompressionRatio, isInSparkInstruction,
			sdcSortType, scaleFactors, preferDeltaEncoding);
	}
}
