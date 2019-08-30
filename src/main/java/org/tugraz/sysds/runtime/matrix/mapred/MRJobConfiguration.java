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
 

package org.tugraz.sysds.runtime.matrix.mapred;

import java.util.ArrayList;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.conf.DMLConfig;
import org.tugraz.sysds.lops.Lop;
import org.tugraz.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.tugraz.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.io.BinaryBlockSerialization;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;

@SuppressWarnings({"deprecation" })
public class MRJobConfiguration 
{
		
	 //internal param: custom deserializer/serializer (usually 30% faster than WritableSerialization)
	public static final boolean USE_BINARYBLOCK_SERIALIZATION = true;
	
	//Job configurations
	
	public static IDSequence seq = new IDSequence();
	
	//matrix indexes to be outputted as final results
	private static final String RESULT_INDEXES_CONFIG="results.indexes";
	private static final String RESULT_DIMS_UNKNOWN_CONFIG="results.dims.unknown";
	
	private static final String INTERMEDIATE_INDEXES_CONFIG="rdiag.indexes";
	
	//output matrices and their formats
	public static final String OUTPUT_MATRICES_DIRS_CONFIG="output.matrices.dirs";
	
	private static final String DIMS_UNKNOWN_FILE_PREFIX = "dims.unknown.file.prefix";
	
	private static final String MMCJ_CACHE_SIZE="mmcj.cache.size";
	
	private static final String DISTCACHE_INPUT_INDICES="distcache.input.indices";
	private static final String DISTCACHE_INPUT_PATHS = "distcache.input.paths";
	
	private static final String SYSTEMDS_LOCAL_TMP_DIR = "systemds.local.tmp.dir";
	
	/*
	 * SystemDS Counter Group names
	 * 
	 * group name for the counters on number of output nonZeros
	 */
	public static final String NUM_NONZERO_CELLS="nonzeros";

	public static final int getMiscMemRequired(JobConf job)
	{
		return job.getInt(MRConfigurationNames.IO_FILE_BUFFER_SIZE, 4096);
	}

	public static void setMMCJCacheSize(JobConf job, long size)
	{
		job.setLong(MMCJ_CACHE_SIZE, size);
	}
	
	public static long getMMCJCacheSize(JobConf job)
	{
		return job.getLong(MMCJ_CACHE_SIZE, 0);
	}

	public static enum ConvertTarget{CELL, BLOCK, WEIGHTEDCELL, CSVWRITE}

	
	/**
	 * Unique working dirs required for thread-safe submission of parallel jobs;
	 * otherwise job.xml and other files might be overridden (in local mode).
	 * 
	 * @param job job configuration
	 */
	public static void setUniqueWorkingDir( JobConf job )
	{
		if( InfrastructureAnalyzer.isLocalMode(job) )
		{
			StringBuilder tmp = new StringBuilder();
			tmp.append( Lop.FILE_SEPARATOR );
			tmp.append( Lop.PROCESS_PREFIX );
			tmp.append( DMLScript.getUUID() );
			tmp.append( Lop.FILE_SEPARATOR );
			tmp.append( seq.getNextID() );
			String uniqueSubdir = tmp.toString();
			
			//unique local dir
			String[] dirlist = job.get(MRConfigurationNames.MR_CLUSTER_LOCAL_DIR,"/tmp").split(",");
			StringBuilder sb2 = new StringBuilder();
			for( String dir : dirlist ) {
				if( sb2.length()>0 )
					sb2.append(",");
				sb2.append(dir);
				sb2.append( uniqueSubdir );
			}
			job.set(MRConfigurationNames.MR_CLUSTER_LOCAL_DIR, sb2.toString() );
			
			//unique system dir 
			job.set(MRConfigurationNames.MR_JOBTRACKER_SYSTEM_DIR, job.get(MRConfigurationNames.MR_JOBTRACKER_SYSTEM_DIR) + uniqueSubdir);
			
			//unique staging dir 
			job.set( MRConfigurationNames.MR_JOBTRACKER_STAGING_ROOT_DIR,  job.get(MRConfigurationNames.MR_JOBTRACKER_STAGING_ROOT_DIR) + uniqueSubdir );
		}
	}
	
	public static String getLocalWorkingDirPrefix(JobConf job)
	{
		return job.get(MRConfigurationNames.MR_CLUSTER_LOCAL_DIR);
	}
	
	public static String getSystemWorkingDirPrefix(JobConf job)
	{
		return job.get(MRConfigurationNames.MR_JOBTRACKER_SYSTEM_DIR);
	}
	
	public static String getStagingWorkingDirPrefix(JobConf job)
	{
		return job.get(MRConfigurationNames.MR_JOBTRACKER_STAGING_ROOT_DIR);
	}
	
	private static byte[] stringArrayToByteArray(String[] istrs) {
		byte[] ret=new byte[istrs.length];
		for(int i=0; i<istrs.length; i++)
			ret[i]=Byte.parseByte(istrs[i]);
		return ret;
	}
	
	public static byte[] getResultIndexes(JobConf job) {
		String[] istrs=job.get(RESULT_INDEXES_CONFIG).split(Instruction.INSTRUCTION_DELIM);
		return stringArrayToByteArray(istrs);
	}
	
	public static byte[] getResultDimsUnknown(JobConf job) {
		String str=job.get(RESULT_DIMS_UNKNOWN_CONFIG);
		if (str==null || str.isEmpty())
			return null;
		String[] istrs=str.split(Instruction.INSTRUCTION_DELIM);
		return stringArrayToByteArray(istrs);
	}
	
	public static byte[] getIntermediateMatrixIndexes(JobConf job) {
		String str=job.get(INTERMEDIATE_INDEXES_CONFIG);
		if(str==null || str.isEmpty())
			return null;
		String[] istrs=str.split(Instruction.INSTRUCTION_DELIM);
		return stringArrayToByteArray(istrs);
	}

	public static void setDimsUnknownFilePrefix(JobConf job, String prefix) {
		job.setStrings(DIMS_UNKNOWN_FILE_PREFIX, prefix);
	}
	
	public static void setupDistCacheInputs(JobConf job, String indices, String pathsString, ArrayList<String> paths) {
		job.set(DISTCACHE_INPUT_INDICES, indices);
		job.set(DISTCACHE_INPUT_PATHS, pathsString);
		Path p = null;
		
		if( !InfrastructureAnalyzer.isLocalMode(job) ) {
			for(String spath : paths) {
				p = new Path(spath);
				
				DistributedCache.addCacheFile(p.toUri(), job);
				DistributedCache.createSymlink(job);
			}
		}
	}
	
	public static String getDistCacheInputIndices(JobConf job) {
		return job.get(DISTCACHE_INPUT_INDICES);
	}
	
	public static boolean deriveRepresentation(InputInfo[] inputInfos) {
		for(InputInfo input: inputInfos)
		{
			if(!(input.inputValueClass==MatrixBlock.class))
			{
				return false;
			}	
		}
		return true;
	}
	
	public static String constructTempOutputFilename() 
	{
		StringBuilder sb = new StringBuilder();
		sb.append(ConfigurationManager.getScratchSpace());
		sb.append(Lop.FILE_SEPARATOR);
		sb.append(Lop.PROCESS_PREFIX);
		sb.append(DMLScript.getUUID());
		sb.append(Lop.FILE_SEPARATOR);
		
		sb.append("TmpOutput"+seq.getNextID());
		
		//old unique dir (no guarantees): 
		//sb.append(Integer.toHexString(new Random().nextInt(Integer.MAX_VALUE))); 
		
		return sb.toString(); 
	}
	
	public static void setSystemDSLocalTmpDir(JobConf job, String dir)
	{
		job.set(SYSTEMDS_LOCAL_TMP_DIR, dir);
	}
	
	public static String getSystemDSLocalTmpDir(JobConf job)
	{
		return job.get(SYSTEMDS_LOCAL_TMP_DIR);
	}
	
	public static void addBinaryBlockSerializationFramework( Configuration job )
	{
		String frameworkList = job.get(MRConfigurationNames.IO_SERIALIZATIONS);
		String frameworkClassBB = BinaryBlockSerialization.class.getCanonicalName();
		job.set(MRConfigurationNames.IO_SERIALIZATIONS, frameworkClassBB+","+frameworkList);
	}
	
	/**
	 * Set all configurations with prefix mapred or mapreduce that exist in the given
	 * DMLConfig into the given JobConf.
	 * 
	 * @param job job configuration
	 * @param config dml configuration
	 */
	public static void setupCustomMRConfigurations( JobConf job, DMLConfig config ) {
		Map<String,String> map = config.getCustomMRConfig();
		for( Entry<String,String> e : map.entrySet() ) {
			job.set(e.getKey(), e.getValue());
		}
	}
}