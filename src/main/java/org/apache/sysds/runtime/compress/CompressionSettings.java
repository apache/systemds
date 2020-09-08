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

import org.apache.sysds.runtime.compress.cocode.PlanningCoCoder.PartitionerType;
import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;

/**
 * Compression Settings class, used as a bundle of parameters inside the Compression framework. See
 * CompressionSettingsBuilder for default non static parameters.
 */
public class CompressionSettings {

	/** Size of the blocks used in a blocked bitmap representation. Note it is one more than Character.MAX_VALUE. */
	public static final int BITMAP_BLOCK_SZ = 65536;

	/**
	 * Sorting of values by physical length helps by 10-20%, especially for serial, while slight performance decrease
	 * for parallel incl multi-threaded, hence not applied for distributed operations (also because compression time +
	 * garbage collection increases)
	 */
	public final boolean sortValuesByLength;

	/**
	 * The sampling ratio used when choosing ColGroups. Note that, default behavior is to use exact estimator if the
	 * number of elements is below 1000.
	 */
	public final double samplingRatio;

	/** Share DDC Dictionaries between ColGroups. */
	public final boolean allowSharedDictionary;

	/**
	 * Transpose input matrix, to optimize performance, this reallocate the matrix to a more cache conscious allocation
	 * for iteration in columns.
	 */
	public final boolean transposeInput;

	/**
	 * Boolean specifying if the OLE and RLE should construct skip to enable skipping large amounts of rows.
	 * (Optimization)
	 */
	public final boolean skipList;

	/** If the seed is -1 then the system used system millisecond time and class hash for seeding. */
	public final int seed;

	/** Boolean specifying if the compression strategy should be investigated and monitored. */
	public final boolean investigateEstimate;

	/** True if lossy compression is enabled */
	public final boolean lossy;

	/** The selected method for column partitioning used in CoCoding compressed columns */
	public final PartitionerType columnPartitioner;

	/** The maximum number of columns CoCoded if the Static CoCoding strategy is selected */
	public final int maxStaticColGroupCoCode;

	/**
	 * Valid Compressions List, containing the ColGroup CompressionTypes that are allowed to be used for the compression
	 * Default is to always allow for Uncompromisable ColGroup.
	 */
	public final EnumSet<CompressionType> validCompressions;

	protected CompressionSettings(double samplingRatio, boolean allowSharedDictionary, boolean transposeInput,
		boolean skipList, int seed, boolean investigateEstimate, boolean lossy,
		EnumSet<CompressionType> validCompressions, boolean sortValuesByLength, PartitionerType columnPartitioner,
		int maxStaticColGroupCoCode) {
		this.samplingRatio = samplingRatio;
		this.allowSharedDictionary = allowSharedDictionary;
		this.transposeInput = transposeInput;
		this.skipList = skipList;
		this.seed = seed;
		this.investigateEstimate = investigateEstimate;
		this.validCompressions = validCompressions;
		this.lossy = lossy;
		this.sortValuesByLength = sortValuesByLength;
		this.columnPartitioner = columnPartitioner;
		this.maxStaticColGroupCoCode = maxStaticColGroupCoCode;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("\n" + super.toString());
		sb.append("\n Valid Compressions: " + validCompressions);
		sb.append("\n DDC1 share dict: " + allowSharedDictionary);
		sb.append("\n Partitioner: " + columnPartitioner);
		sb.append("\n Lossy: " + lossy);
		// If needed for debugging add more fields to the printing.
		return sb.toString();
	}
}
