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

package org.apache.sysml.runtime.compress.cocode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.sysml.runtime.compress.cocode.PlanningCoCoder.GroupableColInfo;

/**
 * Column group partitioning with static distribution heuristic.
 * 
 */
public class ColumnGroupPartitionerStatic extends ColumnGroupPartitioner
{
	private static final int MAX_COL_PER_GROUP = 20;

	@Override
	public List<List<Integer>> partitionColumns(List<Integer> groupCols, HashMap<Integer, GroupableColInfo> groupColsInfo) 
	{
		List<List<Integer>> ret = new ArrayList<>();
		int numParts = (int)Math.ceil((double)groupCols.size()/MAX_COL_PER_GROUP);
		int partSize = (int)Math.ceil((double)groupCols.size()/numParts);
		
		for( int i=0, pos=0; i<numParts; i++, pos+=partSize ) {
			List<Integer> tmp = new ArrayList<>();
			for( int j=0; j<partSize && pos+j<groupCols.size(); j++ )
				tmp.add(groupCols.get(pos+j));
			ret.add(tmp);
		}
		
		return ret;
	}
}
