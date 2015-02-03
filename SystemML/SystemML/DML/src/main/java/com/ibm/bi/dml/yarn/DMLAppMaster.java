/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.yarn;

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

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.runtime.DMLScriptException;

public class DMLAppMaster 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final Log LOG = LogFactory.getLog(DMLAppMaster.class);
	
	private ApplicationId _appId = null;	
	private YarnConfiguration _conf = null;
	
	static
	{
		// for internal debugging only
		if( DMLYarnClientProxy.LDEBUG ) {
			Logger.getLogger("com.ibm.bi.dml.yarn").setLevel((Level) Level.DEBUG);
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
