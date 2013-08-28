package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.hops.Hops.DataGenMethod;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RandInstruction;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class DataGenMapper extends GMRMapper 
implements Mapper<Writable, Writable, Writable, Writable>
{
	private MatrixIndexes indexes=new MatrixIndexes();
	private MatrixBlock block=new MatrixBlock();
	
	
	@Override
	//valueString has to be Text type
	public void map(Writable key, Writable valueString, OutputCollector<Writable, Writable> out,
			Reporter reporter) throws IOException
	{
		if(firsttime)
		{
			cachedReporter=reporter;
			firsttime=false;
		}
		
		long start = System.currentTimeMillis();
		
		//for each represenattive matrix, read the record and apply instructions
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
				long seed=Long.parseLong(params[4]);
				double minValue = randInst.minValue;
				double maxValue = randInst.maxValue;
				double sparsity = randInst.sparsity;
				String pdf = randInst.probabilityDensityFunction;
				
				indexes.setIndexes(blockRowNumber, blockColNumber);
				
				if ( pdf.equalsIgnoreCase("normal") ) { 
					//block.getNormalRandomSparseMatrix(blockRowSize, blockColSize, blockRowSize, blockColSize, sparsity, seed); 
					block.getNormalRandomSparseMatrixOLD(blockRowSize, blockColSize, sparsity, seed); 
				}
				else {
					//block.getRandomSparseMatrix(blockRowSize, blockColSize, blockRowSize, blockColSize, sparsity, minValue, maxValue, seed);
					block.getRandomSparseMatrixOLD(blockRowSize, blockColSize, sparsity, minValue, maxValue, seed);
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
					block.getSequence(from, to, incr);
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
