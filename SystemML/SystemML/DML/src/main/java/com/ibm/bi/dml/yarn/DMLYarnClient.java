/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.yarn;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.v2.util.MRApps;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.ApplicationConstants.Environment;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.ApplicationReport;
import org.apache.hadoop.yarn.api.records.ApplicationSubmissionContext;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.apache.hadoop.yarn.api.records.LocalResourceVisibility;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.api.records.YarnApplicationState;
import org.apache.hadoop.yarn.client.api.YarnClient;
import org.apache.hadoop.yarn.client.api.YarnClientApplication;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.hadoop.yarn.util.Records;

import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

/**
 * NOTES:
 *   * Security: By default, submitted applications are ran as user 'yarn'. 
 *     In order to allow for security and relative filenames on hdfs (/user/<username>/.), 
 *     we can configure the LinuxContainerExecutor in yarn-site.xml, which runs the
 *     application as the user who submits the application.
 *   * SystemML.jar file dependency: We need to submit the SystemML.jar along with the
 *     application. Unfortunately, hadoop jar unpacks the jar such that we dont get a handle
 *     to the original jar filename. We currently parse the constant IBM_JAVA_COMMAND_LINE
 *     to get the jar filename. For robustness, we fall back to repackaging the unzipped files
 *     to a jar if this constant does not exist.  
 * 
 */
public class DMLYarnClient 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private static final Log LOG = LogFactory.getLog(DMLYarnClient.class);
	
	//Internal configuration parameters
	// environment variable to obtain the original jar filename
	private static final String JARFILE_ENV_CONST = "IBM_JAVA_COMMAND_LINE"; 
	// default of 1 core since YARN scheduler does not take the number of cores into account yet 
	private static final int NUM_CORES = 1;  
	// factor for compute virtual memory to request based on given max heap size
	private static final double MEM_FACTOR = 1.5; 
	// default application state report (in milliseconds)
	private static final int APP_STATE_INTERVAL = 200;
	// default application master name
	private static final String APPMASTER_NAME = "SystemML-AM";
	// default application queue
	private static final String APP_QUEUE = "default"; 
	
	
	private String _dmlScript = null;
	private DMLConfig _dmlConfig = null;
	private String[] _args = null; 
	
	//hdfs file names local resources
	private String _hdfsJarFile   = null;
	private String _hdfsDMLScript = null;
	private String _hdfsDMLConfig = null;	
	
	/**
	 * Protected since only supposed to be accessed via proxy in same package.
	 * This is to ensure robustness in case of missing yarn libraries.
	 * 
	 * @param dmlScriptStr
	 * @param conf
	 * @param args
	 */
	protected DMLYarnClient(String dmlScriptStr, DMLConfig conf, String[] args)
	{
		_dmlScript = dmlScriptStr;
		_dmlConfig = conf;
		_args = args;
	}
	
	
	/**
	 * Method to launch the dml yarn app master and execute the given dml script
	 * with the given configuration and jar file.
	 * 
	 * NOTE: on launching the yarn app master, we do not explicitly probe if we
	 *	  are running on a yarn or MR1 cluster. In case of MR1, already the class 
	 *	  YarnConfiguration will not be found and raise a classnotfound. In case of any 
	 *	  exception we fall back to run CP directly in the client process.
	 * 
	 * @return true if dml program successfully executed as yarn app master
	 * @throws IOException 
	 */
	protected boolean launchDMLYarnAppmaster() 
		throws IOException
	{
		boolean ret = false;
		String hdfsWD = null;
		
		try
		{
			Timing time = new Timing(true);
			
			// load yarn configuration
			YarnConfiguration yconf = new YarnConfiguration();
			
			// create yarn client
			YarnClient yarnClient = YarnClient.createYarnClient();
			yarnClient.init(yconf);
			yarnClient.start();
			
			// create application and get the ApplicationID
			YarnClientApplication app = yarnClient.createApplication();
			ApplicationSubmissionContext appContext = app.getApplicationSubmissionContext();
			ApplicationId appId = appContext.getApplicationId();
			LOG.debug("Created application (applicationID: "+appId+")");
			
			// prepare hdfs working directory via ApplicationID
			// copy script, config, jar file to hdfs
			hdfsWD = constructHDFSWorkingDir(_dmlConfig, appId);
			copyResourcesToHdfsWorkingDir(yconf, hdfsWD);
			
			//construct command line argument
			String command = constructAMCommand(_args, _dmlConfig);
			LOG.debug("Constructed application master command: \n"+command);
			
			// set up the container launch context for the application master
			ContainerLaunchContext amContainer = Records.newRecord(ContainerLaunchContext.class);
			amContainer.setCommands( Collections.singletonList(command) );
			amContainer.setLocalResources( constructLocalResourceMap(yconf) );
			amContainer.setEnvironment( constructEnvionmentMap(yconf) );

			// Set up resource type requirements for ApplicationMaster
			int memHeap = _dmlConfig.getIntValue(DMLConfig.YARN_APPMASTERMEM);
			Resource capability = Records.newRecord(Resource.class);
			capability.setMemory( (int)(memHeap*MEM_FACTOR) );
			capability.setVirtualCores( NUM_CORES );
			LOG.debug("Requested application resources: memory="+((int)(memHeap*MEM_FACTOR))+", vcores="+NUM_CORES);

			// Finally, set-up ApplicationSubmissionContext for the application
			appContext.setApplicationName(APPMASTER_NAME); // application name
			appContext.setAMContainerSpec(amContainer);
			appContext.setResource(capability);
			appContext.setQueue(APP_QUEUE); // queue
			LOG.debug("Configured application meta data: name="+APPMASTER_NAME+", queue="+APP_QUEUE);
			
			// submit application (non-blocking)
			yarnClient.submitApplication(appContext);

			// Check application status periodically
			ApplicationReport appReport = yarnClient.getApplicationReport(appId);
			YarnApplicationState appState = appReport.getYarnApplicationState();
			YarnApplicationState oldState = appState;
			LOG.info("Application state: " + appState);
			while( appState != YarnApplicationState.FINISHED
					&& appState != YarnApplicationState.KILLED
					&& appState != YarnApplicationState.FAILED ) 
			{
				Thread.sleep(APP_STATE_INTERVAL); //wait for 200ms
				appReport = yarnClient.getApplicationReport(appId);
				appState = appReport.getYarnApplicationState();
				if( appState != oldState ) {
					oldState = appState;
					LOG.info("Application state: " + appState);
				}
			}
			double appRuntime = (double)(appReport.getFinishTime() - appReport.getStartTime()) / 1000;
			LOG.info( "Application runtime: " + appRuntime + " sec." );
			LOG.info( "Total runtime: " + String.format("%.3f", time.stop()/1000) + " sec.");
			
			ret = true;
		}
		catch(Exception ex)
		{
			LOG.warn("Failed to run DML yarn app master.", ex);
			ret = false;
		}
		finally
		{
			//cleanup working directory
			if( hdfsWD != null )
				MapReduceTool.deleteFileIfExistOnHDFS(hdfsWD);
		}
		
		return ret;
	}

	/**
	 * 
	 * @param conf
	 * @param appId
	 * @return
	 */
	private String constructHDFSWorkingDir(DMLConfig conf, ApplicationId appId)
	{
		StringBuilder sb = new StringBuilder();
		sb.append( conf.getTextValue(DMLConfig.SCRATCH_SPACE) );
		sb.append( Lop.FILE_SEPARATOR );
		sb.append( appId );
		sb.append( Lop.FILE_SEPARATOR );
		return sb.toString();	
	}

	
	/**
	 * 	
	 * @param appId
	 * @throws ParseException
	 * @throws IOException
	 * @throws DMLRuntimeException
	 * @throws InterruptedException 
	 */
	@SuppressWarnings("deprecation")
	private void copyResourcesToHdfsWorkingDir(YarnConfiguration yconf, String hdfsWD ) 
		throws ParseException, IOException, DMLRuntimeException, InterruptedException 
	{
		FileSystem fs = FileSystem.get(yconf);
		
		//create working directory
		MapReduceTool.createDirIfNotExistOnHDFS(hdfsWD, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
		
		//serialize the dml config to HDFS file
		Path confPath = new Path(hdfsWD, "config.xml");
		FSDataOutputStream fout = fs.create(confPath, true);
		fout.writeBytes(_dmlConfig.serializeDMLConfig() + "\n");
		fout.close();
		_hdfsDMLConfig = confPath.makeQualified(fs).toString();
		LOG.debug("DML config written to HDFS file: "+_hdfsDMLConfig+"");

		//serialize the dml script to HDFS file
		Path scriptPath = new Path(hdfsWD, "script.dml");
		FSDataOutputStream fout2 = fs.create(scriptPath, true);
		fout2.writeBytes(_dmlScript);
		fout2.close();
		_hdfsDMLScript = scriptPath.makeQualified(fs).toString();
		LOG.debug("DML script written to HDFS file: "+_hdfsDMLScript+"");
		
		// copy local jar file to HDFS (try to get the original jar filename)
		String fname = getLocalJarFileNameFromEnvConst();
		if( fname == null ){
			//get location of unpacked jar classes and repackage (if required)
			String lclassFile = DMLYarnClient.class.getProtectionDomain().getCodeSource().getLocation().getPath().toString();
			fname = createJar(lclassFile);
		}
		Path srcPath = new Path(fname);
		Path dstPath = new Path(hdfsWD, srcPath.getName());
		FileUtil.copy(FileSystem.getLocal(yconf), srcPath, fs, dstPath, false, true, yconf);
		_hdfsJarFile = dstPath.makeQualified(fs).toString();	
		LOG.debug("Jar file copied from local file: "+srcPath.toString()+" to HDFS file: "+dstPath.toString());
	}
	
	/**
	 * 
	 * @return null if the constant does not exists
	 */
	private String getLocalJarFileNameFromEnvConst()
	{
		String fname = null;
		
		try
		{
			//parse environment constants
			Map<String, String> env = System.getenv();
			if( env.containsKey(JARFILE_ENV_CONST) ){
				String tmp = env.get(JARFILE_ENV_CONST);
				String[] tmpargs = tmp.split(" ");
				for( int i=0; i<tmpargs.length && fname==null; i++ )
					if( tmpargs[i]!=null && tmpargs[i].endsWith("RunJar") )
						fname = tmpargs[i+1];
			}
		}
		catch(Exception ex)
		{
			LOG.warn("Failed to parse environment variables ("+ex.getMessage()+")");
			fname = null; //indicate to use fallback strategy
		}
		
		return fname;
	}
	
	/**
	 * This is our fallback strategy for obtaining our SystemML.jar that we need
	 * to submit as resource for the yarn application. We repackage the unzipped
	 * jar to a temporary jar and later copy it to hdfs.
	 * 
	 * @param dir
	 * @return
	 * @throws IOException
	 * @throws InterruptedException
	 */
	private String createJar( String dir ) 
		throws IOException, InterruptedException
	{
		//construct jar command
		String jarname = dir+"/SystemML.jar";
		String flist = "";
		File fdir = new File(dir);
		File[] tmp = fdir.listFiles();
		for( File ftmp : tmp )
			flist += ftmp.getName()+" ";;
		
		//execute jar command
		String command = "jar cf "+jarname+" "+flist.trim();
		Process child = Runtime.getRuntime().exec(command, null, fdir);		
		int c = 0;
		while ((c = child.getInputStream().read()) != -1)
			System.out.print((char) c);
		while ((c = child.getErrorStream().read()) != -1)
			System.err.print((char) c);
		child.waitFor();
		
		return jarname;
	}
	
	/**
	 * 
	 * @param args
	 * @param conf
	 * @return
	 */
	private String constructAMCommand( String[] args, DMLConfig conf )
	{
		int memHeap = conf.getIntValue(DMLConfig.YARN_APPMASTERMEM);
		StringBuilder command = new StringBuilder();
		command.append("java");
		command.append(" -Xmx"+memHeap+"m");
		command.append(" -Xms"+memHeap+"m");
		command.append(" -Xmn"+(int)(memHeap/10)+"m");
		command.append(' ');
		command.append(DMLAppMaster.class.getName());
	
		//add command line args (modify script and config file path)
		for( int i=0; i<_args.length; i++ )
		{
			String arg = _args[i];
			command.append(' ');
			if( i>0 && _args[i-1].equals("-f") ){
				command.append(_hdfsDMLScript);
				command.append(" -config=" + _hdfsDMLConfig);
			}
			else if( _args[i].startsWith("-config") ){
				//ignore because config added with -f
			}
			else	
				command.append(arg);
		}
	
		//setup stdout and stderr logs 
		command.append(" 1>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stdout");
		command.append(" 2>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stderr");
	
		return command.toString();
	}
	
	/**
	 * 
	 * @param yconf
	 * @param path
	 * @param lpath
	 * @return
	 * @throws IOException
	 */
	private Map<String, LocalResource> constructLocalResourceMap(YarnConfiguration yconf) 
		throws IOException 
	{
		Map<String, LocalResource> rMap = new HashMap<String, LocalResource>();
		Path path = new Path(_hdfsJarFile); 
		
		LocalResource resource = Records.newRecord(LocalResource.class);
		FileStatus jarStat = FileSystem.get(yconf).getFileStatus(path);
		resource.setResource(ConverterUtils.getYarnUrlFromPath(path));
		resource.setSize(jarStat.getLen());
		resource.setTimestamp(jarStat.getModificationTime());
		resource.setType(LocalResourceType.FILE);
		resource.setVisibility(LocalResourceVisibility.PUBLIC);
		
		rMap.put("SystemML.jar", resource);
		return rMap;
	}
	
	/**
	 * 
	 * @param yconf
	 * @return
	 * @throws IOException
	 */
	private Map<String, String> constructEnvionmentMap(YarnConfiguration yconf) 
		throws IOException
	{
		Map<String, String> eMap = new HashMap<String, String>();
		
		//set app master environment
		String classpath = null;
		for (String value : yconf.getStrings(
				YarnConfiguration.YARN_APPLICATION_CLASSPATH,
				YarnConfiguration.DEFAULT_YARN_APPLICATION_CLASSPATH)) 
		{
			if( classpath == null )
				classpath = value.trim();
			else
				classpath = classpath + ":" + value.trim();
		}
		
		eMap.put(Environment.CLASSPATH.name(), classpath);
		MRApps.setClasspath(eMap, yconf);
		
		return eMap;
	}	
}
