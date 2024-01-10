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

import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;

public class LopProperties {
	/** static variable to assign an unique ID to every lop that is created */
	private static IDSequence UniqueLopID =  new IDSequence();
	
	/** 
	 * Execution properties for each lop.
	 * 
	 * execLoc  = in case execType=MR, where does this lop must run in a job (map, reduce, etc.)
	 * compatibleJobs = list of job types in which this lop can be run. It is maintained as a bit vector.
	 * breaksAlignment = does this lop alters the keys ?
	 * isAligner = is this lop mainly used to reorder/sort/align the keys
	 *   
	 */
	protected long ID;
	/** The level in the dag. Specifying when this instruction can be executed. */
	protected int level;
	/** The execution type of this lop node, CP, Spark, GPU, Federated, etc*/
	protected ExecType execType;
	/** If this Lop produce some intermediate that have to be considered in the memory estimations */
	protected boolean producesIntermediateOutput;
	
	public LopProperties() {
		ID = UniqueLopID.getNextID();
		execType = ExecType.INVALID;
		producesIntermediateOutput = false;
	}
	
	public long getID() { return ID; }
	public void setNewID() { ID = UniqueLopID.getNextID(); }
	public int getLevel() { return level; }
	public void setLevel( int l ) { level = l; }
	
	public ExecType getExecType() {
		return execType;
	}

	public void setExecType(ExecType newExecType){
		execType = newExecType;
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

	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" ID: ");
		sb.append(ID);
		sb.append(" Level: ");
		sb.append(level);
		sb.append(" ExecType: ");
		sb.append(execType);
		sb.append(" Intermediate: ");
		sb.append(producesIntermediateOutput);
		return sb.toString();
	}
}
