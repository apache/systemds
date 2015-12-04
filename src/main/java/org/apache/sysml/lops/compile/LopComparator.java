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

package org.apache.sysml.lops.compile;

import java.util.Comparator;

import org.apache.sysml.lops.Lop;


/**
 * 
 * Comparator class used in sorting the LopDAG in topological order. Refer to
 * doTopologicalSort_strict_order() in dml/lops/compile/Dag.java
 * 
 * Topological sort guarantees the following:
 * 
 * 1) All lops with level i appear before any lop with level greater than i
 * (source nodes are at level 0)
 * 
 * 2) Within a given level, nodes are ordered by their ID i.e., by the other in
 * which they are created
 * 
 * compare() method is designed to respect the above two requirements.
 *  
 * @param <N>
 */
public class LopComparator<N extends Lop>
		implements Comparator<N> 
{
	
	@Override
	public int compare(N o1, N o2) {
		if (o1.getLevel() < o2.getLevel())
			return -1; // o1 is less than o2
		else if (o1.getLevel() > o2.getLevel())
			return 1; // o1 is greater than o2
		else {
			if (o1.getID() < o2.getID())
				return -1; // o1 is less than o2
			else if (o1.getID() > o2.getID())
				return 1; // o1 is greater than o2
			else
				throw new RuntimeException("Unexpected error: ID's of two lops are same.");
		}
	}

}
