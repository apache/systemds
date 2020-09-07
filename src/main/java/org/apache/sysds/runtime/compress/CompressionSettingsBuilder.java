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

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.compress.cocode.PlanningCoCoder.PartitionerType;
import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;

/**
 * Builder pattern for Compression Settings. See CompressionSettings for details on values.
 */
public class CompressionSettingsBuilder {
	private double samplingRatio = 1.0;
	private boolean allowSharedDictionary = false;
	private boolean transposeInput = true;
	private boolean skipList = true;
	private int seed = -1;
	private boolean investigateEstimate = false;
	private boolean lossy = false;
	private EnumSet<CompressionType> validCompressions;
	private boolean sortValuesByLength = false;
	private PartitionerType columnPartitioner = PartitionerType.STATIC; // BIN_PACKING or STATIC
	private int maxStaticColGroupCoCode = 1;

	public CompressionSettingsBuilder() {

		DMLConfig conf = ConfigurationManager.getDMLConfig();
		this.lossy = conf.getBooleanValue(DMLConfig.COMPRESSED_LOSSY);
		this.validCompressions = EnumSet.of(CompressionType.UNCOMPRESSED);
		String[] validCompressionsString = conf.getTextValue(DMLConfig.COMPRESSED_VALID_COMPRESSIONS).split(",");
		;
		for(String comp : validCompressionsString) {
			validCompressions.add(CompressionType.valueOf(comp));
		}
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
		this.investigateEstimate = that.investigateEstimate;
		this.validCompressions = EnumSet.copyOf(that.validCompressions);
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
	 * @param transposeInput boolean specifying if the input should be transposed before compression
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setTransposeInput(boolean transposeInput) {
		this.transposeInput = transposeInput;
		return this;
	}

	/**
	 * Specify if the Offset list encoding should utilize skip lists. This increase size of compression but improves
	 * performance in Offset encodings. OLE and RLE.
	 * 
	 * @param skipList a boolean specifying if the skiplist function is enabled
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setSkipList(boolean skipList) {
		this.skipList = skipList;
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
	 * Set if the compression should be investigated while compressing.
	 * 
	 * @param investigateEstimate A boolean specifying it the input should be estimated.
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setInvestigateEstimate(boolean investigateEstimate) {
		this.investigateEstimate = investigateEstimate;
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
	 * Clear all the compression types allowed in the compression. This will only allow the Uncompressed ColGroup type.
	 * Since this is required for operation of the compression
	 * 
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder clearValidCompression() {
		this.validCompressions = EnumSet.of(CompressionType.UNCOMPRESSED);
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
	 * Set the maximum number of columns to CoCode together in the static CoCoding strategy. Compression time increase
	 * with higher numbers.
	 * 
	 * @param maxStaticColGroupCoCode The max selected.
	 * @return The CompressionSettingsBuilder
	 */
	public CompressionSettingsBuilder setmaxStaticColGroupCoCode(int maxStaticColGroupCoCode) {
		this.maxStaticColGroupCoCode = maxStaticColGroupCoCode;
		return this;
	}

	/**
	 * Create the CompressionSettings object to use in the compression.
	 * 
	 * @return The CompressionSettings
	 */
	public CompressionSettings create() {
		return new CompressionSettings(samplingRatio, allowSharedDictionary, transposeInput, skipList, seed,
			investigateEstimate, lossy, validCompressions, sortValuesByLength, columnPartitioner,
			maxStaticColGroupCoCode);
	}
}
