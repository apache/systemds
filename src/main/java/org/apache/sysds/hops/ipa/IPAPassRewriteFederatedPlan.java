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

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.fedplanner.FTypes.FederatedPlanner;
import org.apache.sysds.parser.DMLProgram;

/**
 * This rewrite generates a federated execution plan by estimating and setting costs and the FederatedOutput values of
 * all relevant hops in the DML program.
 * The rewrite is only applied if federated compilation is activated in OptimizerUtils.
 */
public class IPAPassRewriteFederatedPlan extends IPAPass {

	/**
	 * Indicates if an IPA pass is applicable for the current configuration.
	 * The configuration depends on OptimizerUtils.FEDERATED_COMPILATION.
	 *
	 * @param fgraph function call graph
	 * @return true if federated compilation is activated.
	 */
	@Override
	public boolean isApplicable(FunctionCallGraph fgraph) {
		String planner = ConfigurationManager.getDMLConfig()
			.getTextValue(DMLConfig.FEDERATED_PLANNER);
		return OptimizerUtils.FEDERATED_COMPILATION
			|| FederatedPlanner.isCompiled(planner);
	}

	/**
	 * Estimates cost and selects a federated execution plan
	 * by setting the federated output value of each hop in the program.
	 *
	 * @param prog       dml program
	 * @param fgraph     function call graph
	 * @param fcallSizes function call size infos
	 * @return false since the function call graph never has to be rebuilt
	 */
	@Override
	public boolean rewriteProgram(DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes) {
		// obtain planner instance according to config
		String splanner = ConfigurationManager.getDMLConfig()
			.getTextValue(DMLConfig.FEDERATED_PLANNER);
		FederatedPlanner planner = FederatedPlanner.isCompiled(splanner) ?
			FederatedPlanner.valueOf(splanner.toUpperCase()) :
			FederatedPlanner.COMPILE_COST_BASED;
		
		// run planner rewrite with forced federated exec types
		planner.getPlanner().rewriteProgram(prog, fgraph, fcallSizes);
		
		return false;
	}
}
