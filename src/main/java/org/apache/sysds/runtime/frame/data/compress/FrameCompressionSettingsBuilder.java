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

package org.apache.sysds.runtime.frame.data.compress;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.compress.workload.WTreeRoot;

public class FrameCompressionSettingsBuilder {

	private double sampleRatio;
	private int k;
	private WTreeRoot wt;

	public FrameCompressionSettingsBuilder() {
		this.sampleRatio = ConfigurationManager.getDMLConfig().getDoubleValue(DMLConfig.COMPRESSED_SAMPLING_RATIO);
		this.k = 1;
		this.wt = null;
	}

	public FrameCompressionSettingsBuilder wTreeRoot(WTreeRoot wt) {
		this.wt = wt;
		return this;
	}

	public FrameCompressionSettingsBuilder threads(int k) {
		this.k = k;
		return this;
	}

	public FrameCompressionSettingsBuilder sampleRatio(double sampleRatio) {
		this.sampleRatio = sampleRatio;
		return this;
	}

	public FrameCompressionSettings create() {
		return new FrameCompressionSettings(sampleRatio, k, wt);
	}
}
