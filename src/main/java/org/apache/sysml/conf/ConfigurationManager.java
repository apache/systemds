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

package org.apache.sysml.conf;

import org.apache.hadoop.mapred.JobConf;



/**
 * Singleton for accessing the parsed and merged system configuration.
 * 
 * NOTE: parallel execution of multiple DML scripts (in the same JVM) with different configurations  
 *       would require changes/extensions of this class. 
 */
public class ConfigurationManager 
{
	
	private static DMLConfig _conf = null; //read systemml configuration
	private static JobConf _rJob = null; //cached job conf for read-only operations	
	
	static{
		_rJob = new JobConf();
	}
	
	
	/**
	 * 
	 * @param conf
	 */
	public synchronized static void setConfig( DMLConfig conf )
	{
		_conf = conf;
	}
	
	/**
	 * 
	 * @return
	 */
	public synchronized static DMLConfig getConfig()
	{
		return _conf;
	}
	
    /**
     * Returns a cached JobConf object, intended for global use by all operations 
     * with read-only access to job conf. This prevents to read the hadoop conf files
     * over and over again from classpath. However, 
     * 
     * @return
     */
	public static JobConf getCachedJobConf()
	{
		return _rJob;
	}
	
	/**
	 * 
	 * @param job
	 */
	public static void setCachedJobConf(JobConf job) 
	{
		_rJob = job;
	}
}
