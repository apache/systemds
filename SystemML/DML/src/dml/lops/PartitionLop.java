package dml.lops;

import dml.utils.LopsException;
import dml.lops.LopProperties.ExecLocation;
import dml.lops.compile.JobType;
import dml.meta.PartitionParams;
import dml.parser.Expression.*;

public class PartitionLop extends Lops {
	PartitionParams pp ;
	
	public PartitionLop(PartitionParams pp, Lops dataLop, DataType dt, ValueType vt) 
	{ 
		super(Lops.Type.PartitionLop, dt, vt);		
		this.pp = pp ;
		
		if(dataLop != null) 
		{
			this.addInput(dataLop) ;
			dataLop.addOutput(this) ;
		}
		
		/*
		 * This lop can be executed only in PARTITION job.
		 */
		boolean breaksAlignment = true; // TODO: verify
		boolean aligner = false;
		boolean definesMRJob = true;
		lps.addCompatibility(JobType.PARTITION);
		this.lps.setProperties( ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
	}

	@Override
	public String getInstructions(int inputIndex, int outputIndex) throws LopsException 
	{
		throw new LopsException("Should never be invoked for partition lop");

	}

	@Override
	public String getInstructions(int inputIndex1, int inputIndex2,
			int outputIndex) throws LopsException 
	{
		throw new LopsException("Should never be invoked for partition lop");
	}

	@Override
	public String toString() {
		String s = "PartitionLop: " + pp.toString() ;
		return s ;
	}

	public PartitionParams getPartitionParams() {
		return pp;
	}
}