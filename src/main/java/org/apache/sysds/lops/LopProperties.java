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

package org.apache.sysds.lops;

import java.util.ArrayList;

import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;

public class LopProperties 
{
	public enum ExecType { CP, CP_FILE, SPARK, GPU, INVALID, FED }

	// static variable to assign an unique ID to every lop that is created
	private static IDSequence UniqueLopID = null;
	
	static{
		UniqueLopID = new IDSequence();
	}
	
	/** 
	 * Execution properties for each lop.
	 * 
	 * execLoc  = in case execType=MR, where does this lop must run in a job (map, reduce, etc.)
	 * compatibleJobs = list of job types in which this lop can be run. It is maintained as a bit vector.
	 * breaksAlignment = does this lop alters the keys ?
	 * isAligner = is this lop mainly used to reorder/sort/align the keys
	 *   
	 */
	long ID;
	int level;
	ExecType execType;
	boolean producesIntermediateOutput;
	
	public LopProperties() {
		ID = UniqueLopID.getNextID();
		execType = ExecType.INVALID;
		producesIntermediateOutput = false;
	}
	
	public long getID() { return ID; }
	public int getLevel() { return level; }
	public void setLevel( int l ) { level = l; }
	
	public ExecType getExecType() {
		return execType;
	}
	
	public boolean getProducesIntermediateOutput() {
		return producesIntermediateOutput;
	}
	
	public void setProducesIntermediateOutput(boolean pio) {
		producesIntermediateOutput = pio;
	}
	
	/*
	 * Function to compute the node level in the entire Lops DAG. 
	 *   level(v) = max( levels(v.inputs) ) + 1
	 */
	public void setLevel(ArrayList<Lop>  inputs) {
		int tmplevel = -1;
		if ( inputs == null || inputs.isEmpty() )
			tmplevel = 0;
		else {
			// find the max level among all inputs
			for(Lop in : inputs) {
				if(tmplevel < in.getLevel() ) {
					tmplevel = in.getLevel();
				}
			}
			// this.level should be one more than the max
			tmplevel = tmplevel+1;
		}
		setLevel(tmplevel);
	}

	public void setProperties ( ArrayList<Lop> inputs, ExecType et) {
		execType = et;
		setLevel(inputs);
	}
}
