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

package org.apache.sysds.runtime.compress.lib;

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDCSingle;

public class CLALibUtils {

	/**
	 * Helper method to determine if the column groups contains SDC
	 * 
	 * Note that it only returns true, if there is more than one SDC Group.
	 * 
	 * @param groups The ColumnGroups to analyze
	 * @return A Boolean saying it there is >= 2 SDC Groups.
	 */
	protected static boolean containsSDC(List<AColGroup> groups) {
		int count = 0;
		for(AColGroup g : groups) {
			if(g instanceof ColGroupSDC || g instanceof ColGroupSDCSingle) {
				count++;
				if(count > 1)
					break;
			}
		}
		return count > 1;
	}

	/**
	 * Helper method to filter out SDC Groups, to add their common value to the ConstV. This allows exploitation of the
	 * common values in the SDC Groups.
	 * 
	 * @param groups The Column Groups
	 * @param constV The Constant vector to add common values to.
	 * @return The Filtered list of Column groups containing no SDC Groups but only SDCZero groups.
	 */
	protected static List<AColGroup> filterSDCGroups(List<AColGroup> groups, double[] constV) {
		if(constV != null) {
			final List<AColGroup> filteredGroups = new ArrayList<>();
			for(AColGroup g : groups) {
				if(g instanceof ColGroupSDC)
					filteredGroups.add(((ColGroupSDC) g).extractCommon(constV));
				else if(g instanceof ColGroupSDCSingle)
					filteredGroups.add(((ColGroupSDCSingle) g).extractCommon(constV));
				else
					filteredGroups.add(g);
			}
			for(double v : constV)
				if(!Double.isFinite(v))
					return groups;
			
			return filteredGroups;
		}
		else
			return groups;
	}
}
