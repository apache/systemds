package com.ibm.bi.dml.lops.compile;

import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.utils.DMLRuntimeException;


/**
 * This enumeration defines the set of all map-reduce job types. Each job type
 * is associated with following properties:
 * 
 * id - Unique identifier.
 * 
 * name - Job name.
 * 
 * producesIntermediateOutput - set to false if the the job produces an
 * intermediate output that MUST be consumed by subsequent jobs. The
 * intermediate output is NEVER seen by the end user.
 * 
 * emptyInputsAllowed - defines whether or not the job can take an empty input
 * file as an input. Currently, this flag is set to true only for RAND jobs.
 * 
 * allowsSingleShuffleInstruction - set to true if the job allows only a single
 * instruction in the shuffle phase. For example, jobs that perform matrix 
 * multiplication (MMCJ,MMRJ) can perform only one multiplication per job. 
 * Allowing multiple multiplications within a single job complicates 
 * the implementation (due to specialized key-value pairs for each multiplication) 
 * and such a combination can potentially hinder the performance (since these jobs 
 * make use of a lot of memory). Similarly, SORT job can sort a single stream of values.
 * 
 */

public enum JobType {

	/* Add new job types to the following list */
	// (id, name, producesIntermediateOutput, emptyInputsAllowed, allowsSingleShuffleInstruction)
	INVALID			(-1, "INVALID", false, false, false), 
	ANY				(0, "ANY", false, false, false), 
	GMR				(1, "GMR", false, false, false), 
	RAND			(2, "RAND", false, true, false), 
	REBLOCK_TEXT	(3, "REBLOCK_TEXT", false, false, false), 
	REBLOCK_BINARY	(4, "REBLOCK_BINARY", false, false, false), 
	MMCJ			(5, "MMCJ", true, false, true), 
	MMRJ			(6, "MMRJ", false, false, false), 
	COMBINE			(7, "COMBINE", true, false, false), 
	SORT			(8, "SORT", true, false, true), 
	CM_COV			(9, "CM_COV", false, false, false), 
	GROUPED_AGG		(10, "GROUPED_AGG", false, false, false), 
	PARTITION		(11, "PARTITION", false, false, false);
	//REBLOCK		(12, "REBLOCK", true, false);

	/* Following code should not be edited when adding a new job type */

	private final int id;
	private final String name;
	private final boolean producesIntermediateOutput;
	
	private final boolean emptyInputsAllowed;
	
	private final boolean allowsSingleShuffleInstruction;

	JobType(int id, String name, boolean aio, boolean aei, boolean assi) {
		this.id = id;
		this.name = name;
		this.producesIntermediateOutput = aio;
		this.emptyInputsAllowed = aei;
		this.allowsSingleShuffleInstruction = assi;
	}

	public int getId() {
		return id;
	}

	public String getName() {
		return name;
	}

	public boolean producesIntermediateOutput() {
		return producesIntermediateOutput;
	}

	public boolean areEmptyInputsAllowed() {
		return emptyInputsAllowed;
	}

	public boolean allowsSingleShuffleInstruction() {
		return allowsSingleShuffleInstruction;
	}

	public Lops.Type getShuffleLopType() throws DMLRuntimeException {
		if ( allowsSingleShuffleInstruction == false )
			throw new DMLRuntimeException("Shuffle Lop Type is not defined for a job (" + getName() + ") with allowsSingleShuffleInstruction=false.");
		else {
			if ( getName().equals("MMCJ") )
				return Lops.Type.MMCJ;
			else if ( getName().equals("MMRJ") )
				return Lops.Type.MMRJ;
			else if ( getName().equals("SORT") )
				return Lops.Type.SortKeys;
			else 
				throw new DMLRuntimeException("Shuffle Lop Type is not defined for a job (" + getName() + ") that allows a single shuffle instruction.");
		}
	}
	
	public static JobType findJobTypeFromLopType(Lops.Type lt) {
		switch(lt) {
		case RandLop: 		return JobType.RAND;
		
		case Grouping:		return JobType.GMR;
		
		case MMCJ: 			return JobType.MMCJ;
		
		case MMRJ: 			return JobType.MMRJ;
		
		case PartitionLop: 	return JobType.PARTITION;
		
		case SortKeys: 		return JobType.SORT;
		
		case CentralMoment: 
		case CoVariance: 
							return JobType.CM_COV;
		
		case GroupedAgg:	return JobType.GROUPED_AGG;
		
		case CombineBinary: 			
		case CombineTertiary: 			
							return JobType.COMBINE;
		
		default:
			return null;
		}
	}
	
	public boolean isCompatibleWithParentNodes() throws DMLRuntimeException {
		if ( allowsSingleShuffleInstruction == false )
			throw new DMLRuntimeException("isCompatibleWithParentNodes() can not be invoked for a job (" + getName() + ") with allowsSingleShuffleInstruction=false.");
		else {
			if ( getName().equals("MMCJ") || getName().equals("SORT") )
				return false;
			else if ( getName().equals("MMRJ") )
				return true;
			else 
				throw new DMLRuntimeException("Implementation for isCompatibleWithParentNodes() is missing for a job (" + getName() + ") that allows a single shuffle instruction.");
		}
	}
	
	public boolean allowsRecordReaderInstructions() {
		if ( getName().equals("GMR") ) 
			return true;
		else
			return false;
	}
	
	public int getBase() {
		if (id == -1)
			return 0;
		else if (id == 0) {
			// for ANY, return the bit vector with x number of 1's, 
			//   where x = number of actual job types (i.e., excluding INVALID,ANY)
			//System.out.println("ANY --> " + JobType.values().length + ", " + (Math.pow(2, JobType.values().length-2)-1) + ", " + (Math.pow(2,13-2)-1));
			return (int) Math.pow(2, JobType.values().length-2)-1;
		}
		else 
			return (int) Math.pow(2, id-1);
	}

	public JobType findJobTypeById(int id) {
		for (JobType jt : JobType.values()) {
			if (jt.getId() == id)
				return jt;
		}
		return null;
	}

	public JobType findJobTypeByName(String name) {
		for (JobType jt : JobType.values()) {
			if (jt.getName().equalsIgnoreCase(name))
				return jt;
		}
		return null;
	}
	
	public static int getNumJobTypes() {
		return values().length;
	}
}
