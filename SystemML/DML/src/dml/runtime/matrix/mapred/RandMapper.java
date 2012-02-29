package dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.util.MapReduceTool;

public class RandMapper extends GMRMapper 
implements Mapper<Writable, Writable, Writable, Writable>
{
	private Random random=new Random();	
	private MatrixIndexes indexes=new MatrixIndexes();
	private MatrixBlock block=new MatrixBlock();
	
	
	@Override
	//valueString has to be Text type
	public void map(Writable key, Writable valueString, OutputCollector<Writable, Writable> out,
			Reporter reporter) throws IOException
	{
		long start = System.currentTimeMillis();
		
		//for each represenattive matrix, read the record and apply instructions
		for(int i = 0; i < representativeMatrixes.size(); i++)
		{
		//	byte thisMatrix = representativeMatrixes.get(i);
			String[] params = valueString.toString().split(",");
			long blockRowNumber = Long.parseLong(params[0]);
			long blockColNumber = Long.parseLong(params[1]);
			int blockRowSize = Integer.parseInt(params[2]);
			int blockColSize = Integer.parseInt(params[3]);
			long seed=Long.parseLong(params[4]);
			double minValue = rand_instructions.get(i).minValue;
			double maxValue = rand_instructions.get(i).maxValue;
			double sparsity = rand_instructions.get(i).sparsity;
			
			indexes.setIndexes(blockRowNumber, blockColNumber);
			if(sparsity > MatrixBlock.SPARCITY_TURN_POINT)
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
			}
			
			//put the input in the cache
			cachedValues.reset();
			cachedValues.set(rand_instructions.get(i).output, indexes, block);
            
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
