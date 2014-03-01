/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import java.util.ArrayList;

import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;

public class LopProperties 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum ExecType { CP, CP_FILE, MR, INVALID };
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
	
	public LopProperties() {
		ID = UniqueLopID.getNextID();
		execType = ExecType.INVALID;
		execLoc = ExecLocation.INVALID;
		compatibleJobs = JobType.INVALID.getBase();
		breaksAlignment = true;
		isAligner = false;
		definesMRJob = false;
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

	public void setExecLocation(ExecLocation el) {
		execLoc = el;
	}
	
	public void addCompatibility ( JobType jt ) {
		compatibleJobs = compatibleJobs | jt.getBase();
	}
	
	public void removeCompatibility ( JobType jt ) {
		compatibleJobs = compatibleJobs ^ jt.getBase();
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
	
	/*
	 * Function to compute the node level in the entire Lops DAG. 
	 *   level(v) = max( levels(v.inputs) ) + 1
	 */
	public void setLevel(ArrayList<Lop>  inputs) {
		int level = -1;
		if ( inputs == null || inputs.size() == 0)
			level = 0;
		else {
			// find the max level among all inputs
			for(Lop in : inputs) {
				if(level < in.getLevel() ) {
					level = in.getLevel();
				}
			}
			// this.level should be one more than the max
			level = level+1;
		}
		setLevel(level);
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
