/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.lops;

import java.util.ArrayList;

import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;

public class LopProperties 
{
	
	public enum ExecType { CP, CP_FILE, MR, SPARK, INVALID };
	public enum ExecLocation {INVALID, RecordReader, Map, MapOrReduce, MapAndReduce, Reduce, Data, ControlProgram };

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
	ExecLocation execLoc;
	int compatibleJobs;
	boolean breaksAlignment;
	boolean isAligner;
	boolean definesMRJob;
	boolean producesIntermediateOutput;
	
	public LopProperties() {
		ID = UniqueLopID.getNextID();
		execType = ExecType.INVALID;
		execLoc = ExecLocation.INVALID;
		compatibleJobs = JobType.INVALID.getBase();
		breaksAlignment = true;
		isAligner = false;
		definesMRJob = false;
		producesIntermediateOutput = false;
	}
	
	public long getID() { return ID; }
	public int getLevel() { return level; }
	public void setLevel( int l ) { level = l; }
	
	public ExecLocation getExecLocation() {
		return execLoc;
	}
	
	public ExecType getExecType() {
		return execType;
	}
	
	public int getCompatibleJobs() {
		return compatibleJobs;
	}
	
	public boolean getBreaksAlignment() {
		return breaksAlignment;
	}
	
	public boolean getDefinesMRJob() {
		return definesMRJob;
	}
	
	public boolean isAligner()
	{
		return isAligner;
	}
	
	public boolean getProducesIntermediateOutput() {
		return producesIntermediateOutput;
	}

	public void setExecLocation(ExecLocation el) {
		execLoc = el;
	}
	
	public void addCompatibility ( JobType jt ) {
		compatibleJobs = compatibleJobs | jt.getBase();
	}
	
	public void removeCompatibility ( JobType jt ) {
		compatibleJobs = compatibleJobs ^ jt.getBase();
	}
	
	public void removeNonPiggybackableJobs() {
		// Remove compatibility with those jobs which do not allow any "other" instructions 
		for ( JobType jt : JobType.values()) {
			if(jt.allowsNoOtherInstructions()) {
				compatibleJobs = compatibleJobs ^ jt.getBase();
			}
		}
	}
	
	public void setCompatibleJobs(int cj) {
		compatibleJobs = cj;
	}
	
	public void setDefinesMRJob(boolean dmrj) {
		definesMRJob = dmrj;
	}
	
	public void setBreaksAlignment(boolean ba) {
		breaksAlignment = ba;
	}
	
	public void setAligner(boolean align) {
		isAligner = align;
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

	public void setProperties ( ArrayList<Lop> inputs, ExecType et, ExecLocation el, boolean ba, boolean aligner, boolean definesMR ) {
		execType = et;
		execLoc = el;
		breaksAlignment = ba;
		isAligner = aligner;
		definesMRJob = definesMR;
		setLevel(inputs);
	}
	
}
