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


package org.apache.sysml.runtime.matrix.mapred;

import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import org.apache.sysml.hops.Hop.DataGenMethod;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.mr.DataGenMRInstruction;
import org.apache.sysml.runtime.instructions.mr.RandInstruction;
import org.apache.sysml.runtime.matrix.data.LibMatrixDatagen;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.RandomMatrixGenerator;


public class DataGenMapper extends GMRMapper 
implements Mapper<Writable, Writable, Writable, Writable>
{
	
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
