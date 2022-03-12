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

package org.apache.sysds.hops.fedplanner;

import java.util.Map;

import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.fedplanner.FTypes.FType;

public class FederatedPlannerHeuristic extends FederatedPlannerAllFed {
	
	@Override
	protected FType getFederatedOut(Hop hop, Map<Long, FType> fedHops) {
		FType ret = super.getFederatedOut(hop, fedHops); // FedAll
		
		//apply operator-specific heuristics
		if( hop instanceof AggBinaryOp) {
			if( (ret == FType.ROW && hop.getDim2()==1) 
				|| (ret == FType.COL && hop.getDim1()==1) )
			{
				ret = null; //get local vectors
			}
		}
		
		return ret;
	}
}
