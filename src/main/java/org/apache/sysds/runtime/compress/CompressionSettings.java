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

import java.util.List;

import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;

/**
 * Compression Settings class, used as a bundle of parameters inside the Compression framework.
 * See CompressionSettingsBuilder for default non static parameters.
 */
public class CompressionSettings {

	// Sorting of values by physical length helps by 10-20%, especially for serial, while
	// slight performance decrease for parallel incl multi-threaded, hence not applied for
	// distributed operations (also because compression time + garbage collection increases)
	public static final boolean SORT_VALUES_BY_LENGTH = true;

	// The sampling ratio used when choosing ColGroups.
	// Note that, default behavior is to use exact estimator if the number of elements is below 1000.
	public final double samplingRatio;

	// Share DDC Dictionaries between ColGroups.
	// TODO FIX DDC Dictionarie sharing.
	public final boolean allowSharedDDCDictionary;

	// Transpose input matrix, to optimize performance, this reallocate the matrix to a more cache conscious allocation
	// for iteration in columns.
	public final boolean transposeInput;

	// If the seed is -1 then the system used system millisecond time and class hash for seeding.
	public final int seed;

	// Investigate the estimate.
	public final boolean investigateEstimate;

	// Removed the option of LOW_LEVEL_OPT, (only effecting OLE and RLE.)
	// public final boolean LOW_LEVEL_OPT;

	// Valid Compressions List, containing the ColGroup CompressionTypes that are allowed to be used for the compression
	// Default is to always allow for Uncompromisable ColGroup.
	public final List<CompressionType> validCompressions;

	protected CompressionSettings(double samplingRatio, boolean allowSharedDDCDictionary, boolean transposeInput,
		int seed, boolean investigateEstimate, List<CompressionType> validCompressions) {
		this.samplingRatio = samplingRatio;
		this.allowSharedDDCDictionary = allowSharedDDCDictionary;
		this.transposeInput = transposeInput;
		this.seed = seed;
		this.investigateEstimate = investigateEstimate;
		this.validCompressions = validCompressions;
	}



	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("\n" + super.toString());
		sb.append("\n Valid Compressions: " + validCompressions);
		sb.append("\n DDC1 share dict: " + allowSharedDDCDictionary);
		// If needed for debugging add more fields to the printing.
		return sb.toString();
	}
}
