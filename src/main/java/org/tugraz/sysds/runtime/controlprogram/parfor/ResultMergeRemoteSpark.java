/*
 * Modifications Copyright 2019 Graz University of Technology
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.tugraz.sysds.runtime.controlprogram.parfor;


import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.instructions.spark.data.RDDObject;
import org.tugraz.sysds.runtime.instructions.spark.functions.CopyMatrixBlockPairFunction;
import org.tugraz.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.meta.MetaDataFormat;
import org.tugraz.sysds.utils.Statistics;

import java.util.Arrays;

public class ResultMergeRemoteSpark extends ResultMerge
{
	private static final long serialVersionUID = -6924566953903424820L;
	
	private ExecutionContext _ec = null;
	private int  _numMappers = -1;
	private int  _numReducers = -1;
	
	public ResultMergeRemoteSpark(MatrixObject out, MatrixObject[] in, String outputFilename, boolean accum,
		ExecutionContext ec, int numMappers, int numReducers) 
	{
		super(out, in, outputFilename, accum);
		
		_ec = ec;
		_numMappers = numMappers;
		_numReducers = numReducers;
	}

	@Override
	public MatrixObject executeSerialMerge() {
		//graceful degradation to parallel merge
		return executeParallelMerge( _numMappers );
	}
	
	@Override
	public MatrixObject executeParallelMerge(int par) 
	{
		MatrixObject moNew = null; //always create new matrix object (required for nested parallelism)

		if( LOG.isTraceEnabled() )
			LOG.trace("ResultMerge (remote, spark): Execute serial merge for output "
				+_output.hashCode()+" (fname="+_output.getFileName()+")");

		try
		{
			if( _inputs != null && _inputs.length>0 ) {
				//prepare compare
				MetaDataFormat metadata = (MetaDataFormat) _output.getMetaData();
				DataCharacteristics mcOld = metadata.getDataCharacteristics();
				MatrixObject compare = (mcOld.getNonZeros()==0) ? null : _output;
				
				//actual merge
				RDDObject ro = executeMerge(compare, _inputs, mcOld.getRows(), mcOld.getCols(), mcOld.getBlocksize());
				
				//create new output matrix (e.g., to prevent potential export<->read file access conflict
				moNew = new MatrixObject(_output.getValueType(), _outputFName);
				OutputInfo oiOld = metadata.getOutputInfo();
				InputInfo iiOld = metadata.getInputInfo();
				DataCharacteristics mc = new MatrixCharacteristics(mcOld);
				mc.setNonZeros(_isAccum ? -1 : computeNonZeros(_output, Arrays.asList(_inputs)));
				MetaDataFormat meta = new MetaDataFormat(mc,oiOld,iiOld);
				moNew.setMetaData( meta );
				moNew.setRDDHandle( ro );
			}
			else {
				moNew = _output; //return old matrix, to prevent copy
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		
		return moNew;
	}

	@SuppressWarnings("unchecked")
	protected RDDObject executeMerge(MatrixObject compare, MatrixObject[] inputs, long rlen, long clen, int blen)
	{
		String jobname = "ParFor-RMSP";
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		
		SparkExecutionContext sec = (SparkExecutionContext)_ec;
		boolean withCompare = (compare!=null);

		RDDObject ret = null;
		
		//determine degree of parallelism
		int numRed = (int)determineNumReducers(rlen, clen, blen, _numReducers);
		
		//sanity check for empty src files
		if( inputs == null || inputs.length==0  )
			throw new DMLRuntimeException("Execute merge should never be called with no inputs.");
		
		try
		{
			//note: initial implementation via union over all result rdds discarded due to 
			//stack overflow errors with many parfor tasks, and thus many rdds
			
			//Step 1: construct input rdd from all result files of parfor workers
			//a) construct job conf with all files
			InputInfo ii = InputInfo.BinaryBlockInputInfo;
			JobConf job = new JobConf( "test" );
			job.setJobName(jobname);
			job.setInputFormat(ii.inputFormatClass);
			Path[] paths = new Path[ inputs.length ];
			for(int i=0; i<paths.length; i++) {
				//ensure input exists on hdfs (e.g., if in-memory or RDD)
				inputs[i].exportData();
				paths[i] = new Path( inputs[i].getFileName() );
				//update rdd handle to allow lazy evaluation by guarding 
				//against cleanup of temporary result files
				setRDDHandleForMerge(inputs[i], sec);
			}
			FileInputFormat.setInputPaths(job, paths);
			
			//b) create rdd from input files w/ deep copy of keys and blocks
			JavaPairRDD<MatrixIndexes, MatrixBlock> rdd = sec.getSparkContext()
					.hadoopRDD(job, ii.inputFormatClass, ii.inputKeyClass, ii.inputValueClass)
					.mapPartitionsToPair(new CopyMatrixBlockPairFunction(true), true);
			
			//Step 2a: merge with compare
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = null;
			if( withCompare ) {
				JavaPairRDD<MatrixIndexes, MatrixBlock> compareRdd = (JavaPairRDD<MatrixIndexes, MatrixBlock>) 
						sec.getRDDHandleForMatrixObject(compare, InputInfo.BinaryBlockInputInfo);
				
				//merge values which differ from compare values
				ResultMergeRemoteSparkWCompare cfun = new ResultMergeRemoteSparkWCompare(_isAccum);
				out = rdd.groupByKey(numRed) //group all result blocks per key
					.join(compareRdd)        //join compare block and result blocks 
					.mapToPair(cfun);        //merge result blocks w/ compare
			}
			//Step 2b: merge without compare
			else {
				//direct merge in any order (disjointness guaranteed)
				out = _isAccum ?
					RDDAggregateUtils.sumByKeyStable(rdd, false) :
					RDDAggregateUtils.mergeByKey(rdd, false);
			}
			
			//Step 3: create output rdd handle w/ lineage
			ret = new RDDObject(out);
			for(int i=0; i<paths.length; i++)
				ret.addLineageChild(inputs[i].getRDDHandle());
			if( withCompare )
				ret.addLineageChild(compare.getRDDHandle());
		}
		catch( Exception ex ) {
			throw new DMLRuntimeException(ex);
		}
		
		//maintain statistics
		Statistics.incrementNoOfCompiledSPInst();
		Statistics.incrementNoOfExecutedSPInst();
		if( DMLScript.STATISTICS ){
			Statistics.maintainCPHeavyHitters(jobname, System.nanoTime()-t0);
		}
		
		return ret;
	}

	private static int determineNumReducers(long rlen, long clen, int blen, long numRed) {
		//set the number of mappers and reducers 
		long reducerGroups = Math.max(rlen/blen,1) * Math.max(clen/blen, 1);
		return (int)Math.min( numRed, reducerGroups );
	}
	
	@SuppressWarnings("unchecked")
	private static void setRDDHandleForMerge(MatrixObject mo, SparkExecutionContext sec) {
		InputInfo iinfo = InputInfo.BinaryBlockInputInfo;
		JavaSparkContext sc = sec.getSparkContext();
		JavaPairRDD<MatrixIndexes,MatrixBlock> rdd = (JavaPairRDD<MatrixIndexes,MatrixBlock>) 
			sc.hadoopFile( mo.getFileName(), iinfo.inputFormatClass, iinfo.inputKeyClass, iinfo.inputValueClass);
		RDDObject rddhandle = new RDDObject(rdd);
		rddhandle.setHDFSFile(true);
		mo.setRDDHandle(rddhandle);
	}
}
