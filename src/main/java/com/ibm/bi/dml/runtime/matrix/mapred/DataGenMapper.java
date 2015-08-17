/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.mr.DataGenMRInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.RandInstruction;
import com.ibm.bi.dml.runtime.matrix.data.LibMatrixDatagen;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.RandomMatrixGenerator;


public class DataGenMapper extends GMRMapper 
implements Mapper<Writable, Writable, Writable, Writable>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private MatrixIndexes[] indexes=null;
	private MatrixBlock[] block=null;
	
	
	@Override
	//valueString has to be Text type
	public void map(Writable key, Writable valueString, OutputCollector<Writable, Writable> out, Reporter reporter) 
		throws IOException
	{
		cachedReporter=reporter;
		
		long start = System.currentTimeMillis();
		
		//for each representative matrix, read the record and apply instructions
		for(int i = 0; i < representativeMatrixes.size(); i++)
		{
			DataGenMRInstruction genInst = dataGen_instructions.get(i);
			
			if( genInst.getDataGenMethod() == DataGenMethod.RAND ) 
			{
				RandInstruction randInst = (RandInstruction) genInst;
				String[] params = valueString.toString().split(",");
				long blockRowNumber = Long.parseLong(params[0]);
				long blockColNumber = Long.parseLong(params[1]);
				int blockRowSize = Integer.parseInt(params[2]);
				int blockColSize = Integer.parseInt(params[3]);
				long blockNNZ = Integer.parseInt(params[4]);
				long seed=Long.parseLong(params[5]);
				double minValue = randInst.getMinValue();
				double maxValue = randInst.getMaxValue();
				double sparsity = randInst.getSparsity();
				String pdf = randInst.getProbabilityDensityFunction().toLowerCase();
				
				//rand data generation
				try {
					indexes[i].setIndexes(blockRowNumber, blockColNumber);
					
					RandomMatrixGenerator rgen = LibMatrixDatagen.createRandomMatrixGenerator(
																		pdf, blockRowSize, blockColSize, blockRowSize, blockColSize,   
																		sparsity, minValue, maxValue, randInst.getPdfParams() );

					block[i].randOperationsInPlace(rgen, new long[]{blockNNZ}, null, seed); 
				} 
				catch(DMLRuntimeException e) {
					throw new IOException(e);
				}
			}
			else if( genInst.getDataGenMethod() == DataGenMethod.SEQ ) 
			{ 
				String[] params = valueString.toString().split(",");
				long blockRowNumber = Long.parseLong(params[0]);
				long blockColNumber = Long.parseLong(params[1]);
				double from=Double.parseDouble(params[2]);
				double to=Double.parseDouble(params[3]);
				double incr=Double.parseDouble(params[4]);
				
				//sequence data generation
				try {
					indexes[i].setIndexes(blockRowNumber, blockColNumber);
					block[i].seqOperationsInPlace(from, to, incr);
				} 
				catch (DMLRuntimeException e) {
					throw new IOException(e);
				}
			}
			else {
				throw new IOException("Unknown data generation instruction: " + genInst.toString() );
			}
			
			//put the input in the cache
			cachedValues.reset();
			cachedValues.set(genInst.output, indexes[i], block[i]);
            
			//special operations for individual mapp type
			specialOperationsForActualMap(i, out, reporter);
		}
		
		reporter.incrCounter(Counters.MAP_TIME, System.currentTimeMillis() - start);
	}
	
	@Override
	public void configure(JobConf job)
	{
		super.configure(job);
	
		//initialize num_inst matrix indexes and blocks for reuse
		indexes = new MatrixIndexes[representativeMatrixes.size()];
		block = new MatrixBlock[representativeMatrixes.size()];
		for( int i=0; i< representativeMatrixes.size(); i++ ) {
			indexes[i] = new MatrixIndexes();
			block[i] = new MatrixBlock();
		}
	}
}
