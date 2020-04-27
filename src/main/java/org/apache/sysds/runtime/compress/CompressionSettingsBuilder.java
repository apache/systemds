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

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;

/**
 * Builder pattern for Compression Settings.
 * See CompressionSettings for details on values.
 */
public class CompressionSettingsBuilder {
	private double samplingRatio = 0.05;
	private boolean allowSharedDDCDictionary = true;
	private boolean transposeInput = true;
	private int seed = -1;
	private boolean investigateEstimate = false;
	private List<CompressionType> validCompressions = new ArrayList<>();

	public CompressionSettingsBuilder() {
		validCompressions.add(CompressionType.DDC);
		validCompressions.add(CompressionType.OLE);
		validCompressions.add(CompressionType.RLE);
		validCompressions.add(CompressionType.UNCOMPRESSED);
	}
	
	public CompressionSettingsBuilder copySettings(CompressionSettings that){
		this.samplingRatio = that.samplingRatio;
		this.allowSharedDDCDictionary = that.allowSharedDDCDictionary;
		this.transposeInput = that.transposeInput;
		this.seed = that.seed;
		this.investigateEstimate = that.investigateEstimate;
		this.validCompressions = new ArrayList<>(that.validCompressions);
		return this;
	}

	public CompressionSettingsBuilder setSamplingRatio(double samplingRatio) {
		this.samplingRatio = samplingRatio;
		return this;
	}

	public CompressionSettingsBuilder setAllowSharedDDCDictionary(boolean allowSharedDDCDictionary) {
		this.allowSharedDDCDictionary = allowSharedDDCDictionary;
		return this;
	}

	public CompressionSettingsBuilder setTransposeInput(boolean transposeInput) {
		this.transposeInput = transposeInput;
		return this;
	}

	public CompressionSettingsBuilder setSeed(int seed) {
		this.seed = seed;
		return this;
	}

	public CompressionSettingsBuilder setInvestigateEstimate(boolean investigateEstimate) {
		this.investigateEstimate = investigateEstimate;
		return this;
	}

	public CompressionSettingsBuilder setValidCompressions(List<CompressionType> validCompressions) {
		// should always contain Uncompressed as an option.
		if(!validCompressions.contains(CompressionType.UNCOMPRESSED))
			validCompressions.add(CompressionType.UNCOMPRESSED);
		this.validCompressions = validCompressions;
		return this;
	}

	public CompressionSettings create() {
		return new CompressionSettings(samplingRatio, allowSharedDDCDictionary, transposeInput, seed,
			investigateEstimate, validCompressions);
	}
}