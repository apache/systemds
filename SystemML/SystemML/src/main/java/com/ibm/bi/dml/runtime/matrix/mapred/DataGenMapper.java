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
import com.ibm.bi.dml.runtime.instructions.mr.RandInstruction;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;


public class DataGenMapper extends GMRMapper 
implements Mapper<Writable, Writable, Writable, Writable>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private MatrixIndexes indexes=new MatrixIndexes();
	private MatrixBlock block=new MatrixBlock();
	
	
	@Override
	//valueString has to be Text type
	public void map(Writable key, Writable valueString, OutputCollector<Writable, Writable> out,
			Reporter reporter) throws IOException
	{
		cachedReporter=reporter;
		
		long start = System.currentTimeMillis();
		
		//for each representative matrix, read the record and apply instructions
		for(int i = 0; i < representativeMatrixes.size(); i++)
		{
			//DataGenMRInstruction genInst = dataGen_instructions.get(i);
			if ( dataGen_instructions.get(i).getDataGenMethod() == DataGenMethod.RAND ) {
				//	byte thisMatrix = representativeMatrixes.get(i);
				RandInstruction randInst = (RandInstruction) dataGen_instructions.get(i);
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
				String pdf = randInst.getProbabilityDensityFunction();
				
				indexes.setIndexes(blockRowNumber, blockColNumber);
				
				try {
					if ( pdf.equalsIgnoreCase("normal") ) { 
						block.randOperationsInPlace(pdf, blockRowSize, blockColSize, blockRowSize, blockColSize, new long[]{blockNNZ}, sparsity, Double.NaN, Double.NaN, null, seed); 
					}
					else {
						block.randOperationsInPlace(pdf, blockRowSize, blockColSize, blockRowSize, blockColSize, new long[]{blockNNZ}, sparsity, minValue, maxValue, null, seed);
					}
				} catch(DMLRuntimeException e) {
					throw new IOException(e);
				}
			}
			else if ( dataGen_instructions.get(i).getDataGenMethod() == DataGenMethod.SEQ ) { 
				String[] params = valueString.toString().split(",");
				long blockRowNumber = Long.parseLong(params[0]);
				long blockColNumber = Long.parseLong(params[1]);
				double from=Double.parseDouble(params[2]);
				double to=Double.parseDouble(params[3]);
				double incr=Double.parseDouble(params[4]);
				
				try {
					indexes.setIndexes(blockRowNumber, blockColNumber);
					block.seqOperationsInPlace(from, to, incr);
				} catch (DMLRuntimeException e) {
					throw new IOException(e);
				}
			}
			else {
				throw new IOException("Unknown data generation instruction: " + dataGen_instructions.get(i).toString() );
			}
			
			
			// TODO: statiko: check with Yuanyuan if commenting out the following code is ok.
			//       instead, getRandomSparseMatrix() is invoked, as above.
			
			/*if(sparsity > MatrixBlock.SPARCITY_TURN_POINT)
				block.reset(blockRowSize, blockColSize, false);
			else
				block.reset(blockRowSize, blockColSize, true);
			
			double currentValue;
			random.setSeed(seed);
			for(int r = 0; r < blockRowSize; r++)
			{
				for(int c = 0; c < blockColSize; c++)
				{
					if(random.nextDouble() > sparsity)
						continue;
					currentValue = random.nextDouble();//((double) random.nextInt(0, maxRandom) / (double) maxRandom);
					currentValue = (currentValue * (maxValue - minValue) + minValue);
					block.setValue(r, c, currentValue);
				}
			}*/
			
			//put the input in the cache
			cachedValues.reset();
			cachedValues.set(dataGen_instructions.get(i).output, indexes, block);
            
			//System.out.println("generated in Rand: "+indexes +"\n"+block);
			
			//special operations for individual mapp type
			specialOperationsForActualMap(i, out, reporter);
		}
		
		reporter.incrCounter(Counters.MAP_TIME, System.currentTimeMillis() - start);
	}
	
	public void configure(JobConf job)
	{
		super.configure(job);
	//	int id=MapReduceTool.getUniqueMapperId(job, true);
	//	System.out.println("mapper "+ MapReduceTool.getUniqueMapperId(job, true));
	//	System.out.println(job.getNumMapTasks());
	}
}
