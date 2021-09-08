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

package org.apache.sysds.hops.ipa;

import java.util.Map;
import java.util.Map.Entry;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.Compression.CompressConfig;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.runtime.compress.cost.CostEstimatorBuilder;
import org.apache.sysds.runtime.compress.workload.WTreeRoot;
import org.apache.sysds.runtime.compress.workload.WorkloadAnalyzer;

/**
 * This rewrite obtains workload summaries for all hops candidates amenable for compression as a basis for
 * workload-aware compression planning.
 */
public class IPAPassCompressionWorkloadAnalysis extends IPAPass {

	@Override
	public boolean isApplicable(FunctionCallGraph fgraph) {
		return CompressConfig.valueOf(ConfigurationManager.getDMLConfig().getTextValue(DMLConfig.COMPRESSED_LINALG)
			.toUpperCase()) == CompressConfig.WORKLOAD;
	}

	@Override
	public boolean rewriteProgram(DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes) {

		// Set rewrite rule for CLA to false, since we are using workload based planning.
		OptimizerUtils.ALLOW_COMPRESSION_REWRITE = false;

		// Obtain CLA workload analysis for all applicable operators
		Map<Long, WTreeRoot> map = WorkloadAnalyzer.getAllCandidateWorkloads(prog);

		// Add compression instruction to all remaining locations
		for(Entry<Long, WTreeRoot> e : map.entrySet()) {
			final WTreeRoot tree = e.getValue();
			final CostEstimatorBuilder b = new CostEstimatorBuilder(tree);
			final boolean shouldCompress = b.shouldTryToCompress();
			if(LOG.isTraceEnabled())
				LOG.trace("IPAPass Should Compress:\n" + tree + "\n" + b + "\n Should Compress: " + shouldCompress);

			// Filter out compression plans that is known to be bad
			if(shouldCompress) 
				tree.getRoot().setRequiresCompression(tree);

		}

		return map != null;
	}
}
