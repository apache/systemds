/*
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

package org.apache.sysml.runtime.matrix.mapred;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.util.VersionInfo;
import org.apache.sysml.conf.ConfigurationManager;

/**
 * This class provides a central local for used hadoop configuration properties. For portability, we support both hadoop
 * 1.x and 2.x and automatically map to the currently used cluster.
 * 
 */
public abstract class MRConfigurationNames {

	protected static final Log LOG = LogFactory.getLog(MRConfigurationNames.class.getName());

	// non-deprecated properties
	public static final String DFS_DATANODE_DATA_DIR_PERM = "dfs.datanode.data.dir.perm"; // hdfs-default.xml
	public static final String DFS_REPLICATION = "dfs.replication"; // hdfs-default.xml
	public static final String IO_FILE_BUFFER_SIZE = "io.file.buffer.size"; // core-default.xml
	public static final String IO_SERIALIZATIONS = "io.serializations"; // core-default.xml
	public static final String MR_APPLICATION_CLASSPATH = "mapreduce.application.classpath"; // mapred-default.xml
	public static final String MR_CHILD_JAVA_OPTS = "mapred.child.java.opts"; // mapred-default.xml
	public static final String MR_FRAMEWORK_NAME = "mapreduce.framework.name"; // mapred-default.xml
	public static final String MR_JOBTRACKER_STAGING_ROOT_DIR = "mapreduce.jobtracker.staging.root.dir"; // mapred-default.xml
	public static final String MR_TASKTRACKER_GROUP = "mapreduce.tasktracker.group"; // mapred-default.xml
	public static final String YARN_APP_MR_AM_RESOURCE_MB = "yarn.app.mapreduce.am.resource.mb"; // mapred-default.xml

	// deprecated properties replaced by new props, new prop names used for constants
	// see https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-common/DeprecatedProperties.html
	public static final String DFS_BLOCKSIZE; // hdfs-default.xml
	// public static final String DFS_DATANODE_DATA_DIR; // hdfs-default.xml - currently not used
	// public static final String DFS_METRICS_SESSION_ID; // N/A - currently not used
	public static final String DFS_PERMISSIONS_ENABLED; // hdfs-default.xml
	public static final String FS_DEFAULTFS; // core-default.xml
	public static final String MR_CLUSTER_LOCAL_DIR; // mapred-default.xml
	public static final String MR_INPUT_FILEINPUTFORMAT_SPLIT_MAXSIZE; // N/A
	public static final String MR_INPUT_MULTIPLEINPUTS_DIR_FORMATS; // N/A
	public static final String MR_INPUT_MULTIPLEINPUTS_DIR_MAPPERS; // N/A
	public static final String MR_JOB_ID; // N/A
	public static final String MR_JOBTRACKER_ADDRESS; // mapred-default.xml
	public static final String MR_JOBTRACKER_SYSTEM_DIR; // mapred-default.xml
	public static final String MR_MAP_INPUT_FILE; // N/A
	public static final String MR_MAP_INPUT_LENGTH; // N/A
	public static final String MR_MAP_INPUT_START; // N/A
	// NOTE: mapreduce.map.java.opts commented out in mapred-default.xml so as to "not override mapred.child.java.opts"
	public static final String MR_MAP_JAVA_OPTS;
	public static final String MR_MAP_MAXATTEMPTS; // mapred-default.xml
	public static final String MR_MAP_MEMORY_MB; // mapred-default.xml
	public static final String MR_MAP_OUTPUT_COMPRESS; // N/A
	public static final String MR_MAP_OUTPUT_COMPRESS_CODEC; // N/A
	public static final String MR_MAP_SORT_SPILL_PERCENT; // mapred-default.xml
	public static final String MR_REDUCE_INPUT_BUFFER_PERCENT; // N/A
	// NOTE: mapreduce.reduce.java.opts commented out in mapred-default.xml so as to not override mapred.child.java.opts
	public static final String MR_REDUCE_JAVA_OPTS;
	public static final String MR_REDUCE_MEMORY_MB; // mapred-default.xml
	public static final String MR_TASK_ATTEMPT_ID; // N/A
	public static final String MR_TASK_ID; // N/A
	public static final String MR_TASK_IO_SORT_MB; // mapred-default.xml
	public static final String MR_TASK_TIMEOUT; // N/A
	public static final String MR_TASKTRACKER_TASKCONTROLLER; // mapred-default.xml

	// initialize constants based on hadoop version
	static {
		// determine hadoop version
		String hversion = VersionInfo.getBuildVersion();
		boolean hadoop2 = hversion.startsWith("2");
		LOG.debug("Hadoop build version: " + hversion);
		
		// determine mapreduce version
		String mrversion = ConfigurationManager.getCachedJobConf().get(MR_FRAMEWORK_NAME);
		boolean mrv2 = !(mrversion == null || mrversion.equals("classic")); 
		
		//handle hadoop configurations
		if( hadoop2 ) {
			LOG.debug("Using hadoop 2.x configuration properties.");
			DFS_BLOCKSIZE = "dfs.blocksize";
			// DFS_DATANODE_DATA_DIR = "dfs.datanode.data.dir";
			// DFS_METRICS_SESSION_ID = "dfs.metrics.session-id";
			DFS_PERMISSIONS_ENABLED = "dfs.permissions.enabled";
			FS_DEFAULTFS = "fs.defaultFS";
		}
		else {
			LOG.debug("Using hadoop 1.x configuration properties.");
			DFS_BLOCKSIZE = "dfs.block.size";
			// DFS_DATANODE_DATA_DIR = "dfs.data.dir";
			// DFS_METRICS_SESSION_ID = "session.id";
			DFS_PERMISSIONS_ENABLED = "dfs.permissions";
			FS_DEFAULTFS = "fs.default.name";			
		}
			
		//handle mapreduce configurations
		if( mrv2 ) {	
			MR_CLUSTER_LOCAL_DIR = "mapreduce.cluster.local.dir";
			MR_INPUT_FILEINPUTFORMAT_SPLIT_MAXSIZE = "mapreduce.input.fileinputformat.split.maxsize";
			MR_INPUT_MULTIPLEINPUTS_DIR_FORMATS = "mapreduce.input.multipleinputs.dir.formats";
			MR_INPUT_MULTIPLEINPUTS_DIR_MAPPERS = "mapreduce.input.multipleinputs.dir.mappers";
			MR_JOB_ID = "mapreduce.job.id";
			MR_JOBTRACKER_ADDRESS = "mapreduce.jobtracker.address";
			MR_JOBTRACKER_SYSTEM_DIR = "mapreduce.jobtracker.system.dir";
			MR_MAP_INPUT_FILE = "mapreduce.map.input.file";
			MR_MAP_INPUT_LENGTH = "mapreduce.map.input.length";
			MR_MAP_INPUT_START = "mapreduce.map.input.start";
			MR_MAP_JAVA_OPTS = "mapreduce.map.java.opts";
			MR_MAP_MAXATTEMPTS = "mapreduce.map.maxattempts";
			MR_MAP_MEMORY_MB = "mapreduce.map.memory.mb";
			MR_MAP_OUTPUT_COMPRESS = "mapreduce.map.output.compress";
			MR_MAP_OUTPUT_COMPRESS_CODEC = "mapreduce.map.output.compress.codec";
			MR_MAP_SORT_SPILL_PERCENT = "mapreduce.map.sort.spill.percent";
			MR_REDUCE_INPUT_BUFFER_PERCENT = "mapreduce.reduce.input.buffer.percent";
			MR_REDUCE_JAVA_OPTS = "mapreduce.reduce.java.opts";
			MR_REDUCE_MEMORY_MB = "mapreduce.reduce.memory.mb";
			MR_TASK_ATTEMPT_ID = "mapreduce.task.attempt.id";
			MR_TASK_ID = "mapreduce.task.id";
			MR_TASK_IO_SORT_MB = "mapreduce.task.io.sort.mb";
			MR_TASK_TIMEOUT = "mapreduce.task.timeout";
			MR_TASKTRACKER_TASKCONTROLLER = "mapreduce.tasktracker.taskcontroller";
		} 
		else { // mrv1
			MR_CLUSTER_LOCAL_DIR = "mapred.local.dir";
			MR_INPUT_FILEINPUTFORMAT_SPLIT_MAXSIZE = "mapred.max.split.size";
			MR_INPUT_MULTIPLEINPUTS_DIR_FORMATS = "mapred.input.dir.formats";
			MR_INPUT_MULTIPLEINPUTS_DIR_MAPPERS = "mapred.input.dir.mappers";
			MR_JOB_ID = "mapred.job.id";
			MR_JOBTRACKER_ADDRESS = "mapred.job.tracker";
			MR_JOBTRACKER_SYSTEM_DIR = "mapred.system.dir";
			MR_MAP_INPUT_FILE = "map.input.file";
			MR_MAP_INPUT_LENGTH = "map.input.length";
			MR_MAP_INPUT_START = "map.input.start";
			MR_MAP_JAVA_OPTS = "mapred.map.child.java.opts";
			MR_MAP_MAXATTEMPTS = "mapred.map.max.attempts";
			MR_MAP_MEMORY_MB = "mapred.job.map.memory.mb";
			MR_MAP_OUTPUT_COMPRESS = "mapred.compress.map.output";
			MR_MAP_OUTPUT_COMPRESS_CODEC = "mapred.map.output.compression.codec";
			MR_MAP_SORT_SPILL_PERCENT = "io.sort.spill.percent";
			MR_REDUCE_INPUT_BUFFER_PERCENT = "mapred.job.reduce.input.buffer.percent";
			MR_REDUCE_JAVA_OPTS = "mapred.reduce.child.java.opts";
			MR_REDUCE_MEMORY_MB = "mapred.job.reduce.memory.mb";
			MR_TASK_ATTEMPT_ID = "mapred.task.id";
			MR_TASK_ID = "mapred.tip.id";
			MR_TASK_IO_SORT_MB = "io.sort.mb";
			MR_TASK_TIMEOUT = "mapred.task.timeout";
			MR_TASKTRACKER_TASKCONTROLLER = "mapred.task.tracker.task-controller";
		}
	}
}
