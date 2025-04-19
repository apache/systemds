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

public class FTypes
{
	public enum FederatedPlanner {
		NONE,
		RUNTIME,
		COMPILE_FED_ALL,
		COMPILE_FED_HEURISTIC,
		COMPILE_COST_BASED;
		public AFederatedPlanner getPlanner() {
			switch( this ) {
				case COMPILE_FED_ALL:
					return new FederatedPlannerFedAll();
				case COMPILE_FED_HEURISTIC:
					return new FederatedPlannerFedHeuristic();
				case COMPILE_COST_BASED:
					return new FederatedPlannerFedCostBased();
				case NONE:
				case RUNTIME:
				default:
					return null;
			}
		}
		public boolean isCompiled() {
			return this != NONE && this != RUNTIME;
		}
		public static boolean isCompiled(String planner) {
			return planner != null 
				&& FederatedPlanner.valueOf(planner.toUpperCase()).isCompiled();
		}
	}
	
	public enum FPartitioning {
		ROW,   //row partitioned, groups of entire rows
		COL,   //column partitioned, groups of entire columns
		MIXED, //arbitrary rectangles
		NONE,  //entire data in a location
	}

	public enum FReplication {
		NONE,    //every data item in a separate location
		FULL,    //every data item at every location
		OVERLAP, //every data item partially at every location, w/ addition as aggregation method
	}

	public enum FType {
		ROW(FPartitioning.ROW, FReplication.NONE),
		COL(FPartitioning.COL, FReplication.NONE),
		FULL(FPartitioning.NONE, FReplication.NONE),
		BROADCAST(FPartitioning.NONE, FReplication.FULL),
		PART(FPartitioning.NONE, FReplication.OVERLAP),
		OTHER(FPartitioning.MIXED, FReplication.NONE);

		private final FPartitioning _partType;
		private final FReplication _repType;

		private FType(FPartitioning ptype, FReplication rtype) {
			_partType = ptype;
			_repType = rtype;
		}

		public boolean isRowPartitioned() {
			return _partType == FPartitioning.ROW
				|| (_partType == FPartitioning.NONE
				&& !(_repType == FReplication.OVERLAP));
		}

		public boolean isColPartitioned() {
			return _partType == FPartitioning.COL
				|| (_partType == FPartitioning.NONE
				&& !(_repType == FReplication.OVERLAP));
		}

		public FPartitioning getPartType() {
			return this._partType;
		}

		public boolean isType(FType t) {
			switch(t) {
				case ROW:
					return isRowPartitioned();
				case COL:
					return isColPartitioned();
				case FULL:
				case OTHER:
				default:
					return t == this;
			}
		}
	}

	// Alignment Check Type
	public enum AlignType {
		FULL, // exact matching dimensions of partitions on the same federated worker
		ROW, // matching rows of partitions on the same federated worker
		COL, // matching columns of partitions on the same federated worker
		FULL_T, // matching dimensions with transposed dimensions of partitions on the same federated worker
		ROW_T, // matching rows with columns of partitions on the same federated worker
		COL_T; // matching columns with rows of partitions on the same federated worker

		public boolean isTransposed() {
			return (this == FULL_T || this == ROW_T || this == COL_T);
		}
		public boolean isFullType() {
			return (this == FULL || this == FULL_T);
		}
		public boolean isRowType() {
			return (this == ROW || this == ROW_T);
		}
		public boolean isColType() {
			return (this == COL || this == COL_T);
		}
	}
}
