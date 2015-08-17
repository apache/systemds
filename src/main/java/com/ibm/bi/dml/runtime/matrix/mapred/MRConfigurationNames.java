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


package com.ibm.bi.dml.runtime.matrix.mapred;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.util.VersionInfo;


/**
 * This class provides a central local for used hadoop configuration properties.
 * For portability, we support both hadoop 1.x and 2.x and automatically map to 
 * the currently used cluster.
 * 
 */
public abstract class MRConfigurationNames 
{

	
	protected static final Log LOG = LogFactory.getLog(MRConfigurationNames.class.getName());
	
	//name definitions
	public static final String INVALID = "null";
	public static String DFS_SESSION_ID = INVALID;
	public static String DFS_BLOCK_SIZE = INVALID;
	public static String DFS_PERMISSIONS = INVALID;

	//initialize to used cluster 
	static{
		
		//determine hadoop version
		//e.g., 2.0.4-alpha from 0a11e32419bd4070f28c6d96db66c2abe9fd6d91 by jenkins source checksum f3c1bf36ae3aa5a6f6d3447fcfadbba
		String version = VersionInfo.getBuildVersion();
		boolean hadoopVersion2 = version.startsWith("2");
		LOG.debug("Hadoop build version: "+version);
		
		if( hadoopVersion2 )
		{
			LOG.debug("Using hadoop 2.x configuration properties.");
			DFS_SESSION_ID  = "dfs.metrics.session-id";
			DFS_BLOCK_SIZE  = "dfs.blocksize";
			DFS_PERMISSIONS = "dfs.permissions.enabled";
		}
		else //any older version
		{
			LOG.debug("Using hadoop 1.x configuration properties.");
			DFS_SESSION_ID  = "session.id";
			DFS_BLOCK_SIZE  = "dfs.block.size";
			DFS_PERMISSIONS = "dfs.permissions";
		}
	}
}
