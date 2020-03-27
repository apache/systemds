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

package org.apache.sysds.hops.codegen.opt;

import java.util.ArrayList;
import java.util.Map.Entry;

import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.codegen.template.CPlanMemoTable;
import org.apache.sysds.hops.codegen.template.CPlanMemoTable.MemoTableEntry;

import java.util.List;

/**
 * This plan selection heuristic aims for maximal fusion, which
 * potentially leads to overlapping fused operators and thus,
 * redundant computation but with a minimal number of materialized
 * intermediate results.
 * 
 */
public class PlanSelectionFuseAll extends PlanSelection
{	
	@Override
	public void selectPlans(CPlanMemoTable memo, ArrayList<Hop> roots) {
		//pruning and collection pass
		for( Hop hop : roots )
			rSelectPlansFuseAll(memo, hop, null, null);
		
		//take all distinct best plans
		for( Entry<Long, List<MemoTableEntry>> e : getBestPlans().entrySet() )
			memo.setDistinct(e.getKey(), e.getValue());
	}
}
