/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops.compile;

import com.ibm.bi.dml.hops.Hop.FileFormatTypes;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.runtime.DMLRuntimeException;


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

public enum JobType 
{
	/* Add new job types to the following list */
	// (id, name, producesIntermediateOutput, emptyInputsAllowed, allowsSingleShuffleInstruction, allowsNoOtherInstructions)
	INVALID			(-1, "INVALID", false, false, false, false), 
	ANY				(0, "ANY", false, false, false, false), 
	GMR				(1, "GMR", false, false, false, false), 
	DATAGEN			(2, "DATAGEN", false, true, false, false), 
	REBLOCK			(3, "REBLOCK", false, false, false, false), 
	MMCJ			(4, "MMCJ", true, false, true, false), 
	MMRJ			(5, "MMRJ", false, false, false, false), 
	COMBINE			(6, "COMBINE", true, false, false, true), 
	SORT			(7, "SORT", true, false, true, true),  			// allows only "InstructionsBeforeSort" and nothing else. 
	CM_COV			(8, "CM_COV", false, false, false, false),  	// allows only instructions in the mapper 
	GROUPED_AGG		(9, "GROUPED_AGG", false, false, false, false), 
	//PARTITION		(10, "PARTITION", false, false, false, true),	// MB: meta learning removed
	DATA_PARTITION	(11, "DATAPARTITION", false, false, false, true),
	CSV_REBLOCK		(12, "CSV_REBLOCK", false, false, false, false),
	CSV_WRITE		(13, "CSV_WRITE", false, false, false, true);

	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/* Following code should not be edited when adding a new job type */

	private final int id;
	private final String name;
	private final boolean producesIntermediateOutput;
	
	private final boolean emptyInputsAllowed;
	
	private final boolean allowsSingleShuffleInstruction;

	/**
	 * Indicates whether a job can piggyback "other" operations. 
	 * For example, COMBINE job can only piggyback multiple combine operators but can not perform any other operations.
	 */
	private final boolean allowsNoOtherInstructions;
	
	JobType(int id, String name, boolean aio, boolean aei, boolean assi, boolean anoi) {
		this.id = id;
		this.name = name;
		this.producesIntermediateOutput = aio;
		this.emptyInputsAllowed = aei;
		this.allowsSingleShuffleInstruction = assi;
		this.allowsNoOtherInstructions = anoi;
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

	public boolean allowsNoOtherInstructions() {
		return allowsNoOtherInstructions;
	}

	public Lop.Type getShuffleLopType() throws DMLRuntimeException {
		if ( allowsSingleShuffleInstruction == false )
			throw new DMLRuntimeException("Shuffle Lop Type is not defined for a job (" + getName() + ") with allowsSingleShuffleInstruction=false.");
		else {
			if ( getName().equals("MMCJ") )
				return Lop.Type.MMCJ;
			else if ( getName().equals("MMRJ") )
				return Lop.Type.MMRJ;
			else if ( getName().equals("SORT") )
				return Lop.Type.SortKeys;
			else 
				throw new DMLRuntimeException("Shuffle Lop Type is not defined for a job (" + getName() + ") that allows a single shuffle instruction.");
		}
	}
	
	public static JobType findJobTypeFromLop(Lop node) {
		Lop.Type lt = node.getType();
		switch(lt) {
		case DataGen: 		return JobType.DATAGEN;
		
		case ReBlock:		return JobType.REBLOCK;
		
		case Grouping:		return JobType.GMR;
		
		case MMCJ: 			return JobType.MMCJ;
		
		case MMRJ: 			return JobType.MMRJ;
		
		case MMTSJ: 		return JobType.GMR;
		
		case SortKeys: 		return JobType.SORT;
		
		case CentralMoment: 
		case CoVariance: 
							return JobType.CM_COV;
		
		case GroupedAgg:	return JobType.GROUPED_AGG;
		
		case CombineBinary: 			
		case CombineTertiary: 			
							return JobType.COMBINE;
		
		case DataPartition:	return JobType.DATA_PARTITION;
		
		case CSVReBlock:	return JobType.CSV_REBLOCK;
		
		case Data:
			/*
			 * Only Write LOPs with external data formats (except MatrixMarket) produce MR Jobs
			 */
			FileFormatTypes fmt = ((Data) node).getFileFormatType();
			if ( fmt == FileFormatTypes.CSV )
				return JobType.CSV_WRITE;
			else
				return null;
			
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
