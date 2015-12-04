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

package org.apache.sysml.yarn;

import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.yarn.api.ApplicationConstants.Environment;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.client.api.AMRMClient;
import org.apache.hadoop.yarn.client.api.AMRMClient.ContainerRequest;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.runtime.DMLScriptException;

public class DMLAppMaster 
{
	
	private static final Log LOG = LogFactory.getLog(DMLAppMaster.class);
	
	private ApplicationId _appId = null;	
	private YarnConfiguration _conf = null;
	
	static
	{
		// for internal debugging only
		if( DMLYarnClientProxy.LDEBUG ) {
			Logger.getLogger("org.apache.sysml.yarn").setLevel((Level) Level.DEBUG);
		}
	}
	
	/**
	 * 
	 * @param args
	 * @throws YarnException 
	 * @throws IOException 
	 */
	public void runApplicationMaster( String[] args ) 
		throws YarnException, IOException
	{
		_conf = new YarnConfiguration();
		
		//obtain application ID
		String containerIdString = System.getenv(Environment.CONTAINER_ID.name());
	    ContainerId containerId = ConverterUtils.toContainerId(containerIdString);
	    _appId = containerId.getApplicationAttemptId().getApplicationId();
	    LOG.info("SystemML appplication master (applicationID: " + _appId + ")");
	    
	    //initialize clients to ResourceManager
 		AMRMClient<ContainerRequest> rmClient = AMRMClient.createAMRMClient();
 		rmClient.init(_conf);
 		rmClient.start();
		
		//register with ResourceManager
		rmClient.registerApplicationMaster("", 0, ""); //host, port for rm communication
		LOG.debug("Registered the SystemML application master with resource manager");
		
		//start status reporter to ResourceManager
		DMLAppMasterStatusReporter reporter = new DMLAppMasterStatusReporter(rmClient, 10000);
		reporter.start();
		LOG.debug("Started status reporter (heartbeat to resource manager)");
		
		
		//set DMLscript app master context
 		DMLScript.setActiveAM();
 		
		//parse input arguments
		String[] otherArgs = new GenericOptionsParser(_conf, args).getRemainingArgs();
		
		//run SystemML CP
		FinalApplicationStatus status = null;
		try
		{
			//core dml script execution (equivalent to non-AM runtime)
			boolean success = DMLScript.executeScript(_conf, otherArgs);
			
			if( success )
				status = FinalApplicationStatus.SUCCEEDED;
			else
				status = FinalApplicationStatus.FAILED;
		}
		catch(DMLScriptException ex)
		{
			LOG.error( DMLYarnClient.APPMASTER_NAME+": Failed to executed DML script due to stop call:\n\t" + ex.getMessage() );
			status = FinalApplicationStatus.FAILED;
			writeMessageToHDFSWorkingDir( ex.getMessage() );
		}
		catch(Exception ex)
		{
			LOG.error( DMLYarnClient.APPMASTER_NAME+": Failed to executed DML script.", ex );
			status = FinalApplicationStatus.FAILED;
		}
		finally
		{
			//stop periodic status reports
			reporter.stopStatusReporter();
			LOG.debug("Stopped status reporter");
				
			//unregister resource manager client
			rmClient.unregisterApplicationMaster(status, "", "");
			LOG.debug("Unregistered the SystemML application master");
		}
	}
	
	/**
	 * 
	 * @param msg
	 */
	private void writeMessageToHDFSWorkingDir(String msg)
	{
		//construct working directory (consistent with client)
		DMLConfig conf = ConfigurationManager.getConfig();
		String hdfsWD = DMLAppMasterUtils.constructHDFSWorkingDir(conf, _appId);
		Path msgPath = new Path(hdfsWD, DMLYarnClient.DML_STOPMSG_NAME);
		
		//write given message to hdfs
		try {
			FileSystem fs = FileSystem.get(_conf);
			FSDataOutputStream fout = fs.create(msgPath, true);
			fout.writeBytes( msg );
			fout.close();
			LOG.debug("Stop message written to HDFS file: "+msgPath );
		}
		catch(Exception ex) {
			LOG.error("Failed to write stop message to HDFS file: "+msgPath, ex);
		}
	}
	
	/**
	 * Main entrance for starting the SystemML app master.
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) 
		throws Exception 
	{
		try
		{
			DMLAppMaster am = new DMLAppMaster();
			am.runApplicationMaster( args );
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
	}
}
