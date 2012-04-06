package dml.lops;

import dml.lops.compile.JobType;

public class LopProperties {
	public enum ExecType { CP, MR, INVALID };
	public enum ExecLocation {INVALID, RecordReader, Map, MapOrReduce, MapAndReduce, Reduce, Data, ControlProgram };

	// static variable to assign an unique ID to every lop that is created
	private static int UniqueLopID = 0;
	
	/** 
	 * Execution properties for each lop.
	 * 
	 * execLoc  = in case execType=MR, where does this lop must run in a job (map, reduce, etc.)
	 * compatibleJobs = list of job types in which this lop can be run. It is maintained as a bit vector.
	 * breaksAlignment = does this lop alters the keys ?
	 * isAligner = is this lop mainly used to reorder/sort/align the keys
	 *   
	 */
	int ID, level;
	ExecType execType;
	ExecLocation execLoc;
	int compatibleJobs;
	boolean breaksAlignment;
	boolean isAligner;
	boolean definesMRJob;
	
	private static int getNextLopID() {
		return ++UniqueLopID;
	}
	
	public LopProperties() {
		ID = getNextLopID();
		execType = ExecType.INVALID;
		execLoc = ExecLocation.INVALID;
		compatibleJobs = JobType.INVALID.getBase();
		breaksAlignment = true;
		isAligner = false;
		definesMRJob = false;
	}
	
	public int getID() { return ID; }
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
	
	public void setProperties ( ExecType et, ExecLocation el, boolean ba, boolean aligner, boolean definesMR ) {
		execType = et;
		execLoc = el;
		breaksAlignment = ba;
		isAligner = aligner;
		definesMRJob = definesMR;
	}
	
}
