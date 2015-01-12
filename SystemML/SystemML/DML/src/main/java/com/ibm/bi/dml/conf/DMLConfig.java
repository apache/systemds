/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.conf;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.StringWriter;
import java.util.HashMap;

import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;


public class DMLConfig 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	public static final String DEFAULT_SYSTEMML_CONFIG_FILEPATH = "./SystemML-config.xml";
	
	private static final Log LOG = LogFactory.getLog(DMLConfig.class.getName());
	
	// external names of configuration properties 
	// (single point of change for all internal refs)
	public static final String LOCAL_TMP_DIR        = "localtmpdir";
	public static final String SCRATCH_SPACE        = "scratch";
	public static final String OPTIMIZATION_LEVEL   = "optlevel";	
	public static final String NUM_REDUCERS         = "numreducers";
	public static final String JVM_REUSE            = "jvmreuse";
	public static final String DEFAULT_BLOCK_SIZE   = "defaultblocksize"; 	
	public static final String YARN_APPMASTER       = "dml.yarn.appmaster"; 	
	public static final String YARN_APPMASTERMEM    = "dml.yarn.appmaster.mem"; 
	public static final String YARN_MAPREDUCEMEM    = "dml.yarn.mapreduce.mem"; 
	public static final String NUM_MERGE_TASKS      = "NumMergeTasks";
	public static final String NUM_SOW_THREADS      = "NumberOfSowThreads";
	public static final String NUM_REAP_THREADS     = "NumberOfReapThreads";
	public static final String SOWER_WAIT_INTERVAL  = "SowerWaitInterval";
	public static final String REAPER_WAIT_INTERVAL = "ReaperWaitInterval";
	public static final String NIMBLE_SCRATCH       = "NimbleScratch";

	//internal config
	public static final String DEFAULT_SHARED_DIR_PERMISSION = "777"; //for local fs and DFS
	public static String LOCAL_MR_MODE_STAGING_DIR = null;
	
	//configuration default values
	private static HashMap<String, String> _defaultVals = null;

    private String config_file_name = null;
	private Element xml_root = null;
	
	static
	{
		_defaultVals = new HashMap<String, String>();
		_defaultVals.put(LOCAL_TMP_DIR,        "/tmp/systemml" );
		_defaultVals.put(SCRATCH_SPACE,        "scratch_space" );
		_defaultVals.put(OPTIMIZATION_LEVEL,   "2" );
		_defaultVals.put(NUM_REDUCERS,         "10" );
		_defaultVals.put(JVM_REUSE,            "false" );
		_defaultVals.put(DEFAULT_BLOCK_SIZE,   "1000" );
		_defaultVals.put(YARN_APPMASTER,       "false" );
		_defaultVals.put(YARN_APPMASTERMEM,    "2048" );
		_defaultVals.put(YARN_MAPREDUCEMEM,    "-1" );
		_defaultVals.put(NUM_MERGE_TASKS,      "4" );
		_defaultVals.put(NUM_SOW_THREADS,      "1" );
		_defaultVals.put(NUM_REAP_THREADS,     "1" );
		_defaultVals.put(SOWER_WAIT_INTERVAL,  "1000" );
		_defaultVals.put(REAPER_WAIT_INTERVAL, "1000" );
		_defaultVals.put(NIMBLE_SCRATCH,       "nimbleoutput" );	
	}
	
	public DMLConfig()
	{
		
	}
	
	/**
	 * 
	 * @param fileName
	 * @throws ParseException
	 */
	public DMLConfig(String fileName) 
		throws ParseException
	{
		this( fileName, false );
	}
	
	/**
	 * 
	 * @param fileName
	 * @param silent
	 * @throws ParseException
	 */
	public DMLConfig(String fileName, boolean silent) 
		throws ParseException
	{
		config_file_name = fileName;
		try {
			parseConfig();
		}
		catch (Exception e){
		    //log error, since signature of generated ParseException doesn't allow to pass it 
			if( !silent )
				LOG.error("Failed to parse DML config file ",e);
			throw new ParseException("ERROR: error parsing DMLConfig file " + fileName);
		}
		
		LOCAL_MR_MODE_STAGING_DIR = getTextValue(LOCAL_TMP_DIR) + "/hadoop/mapred/staging";
	}
	
	
	public String getConfig_file_name() 
	{
		return config_file_name;
	}
	
	public DMLConfig( Element root )
	{
		xml_root = root;
	}
	
	public void merge(DMLConfig otherConfig) 
		throws ParseException
	{
		if (otherConfig == null) 
			return;
	
		try {
			// for each element in otherConfig, either overwrite existing value OR add to defaultConfig
			NodeList otherConfigNodeList = otherConfig.xml_root.getChildNodes();
			if (otherConfigNodeList != null && otherConfigNodeList.getLength() > 0){
				for (int i=0; i<otherConfigNodeList.getLength(); i++){
					org.w3c.dom.Node optionalConfigNode = otherConfigNodeList.item(i);
					
					if (optionalConfigNode.getNodeType() == org.w3c.dom.Node.ELEMENT_NODE){
					
						// try to find optional config node in default config node
						String paramName = optionalConfigNode.getNodeName();
						String paramValue = ((Element)optionalConfigNode).getFirstChild().getNodeValue();
					
						if (this.xml_root.getElementsByTagName(paramName) != null)
							LOG.info("Updating " + paramName + " with value " + paramValue);
						else 
							LOG.info("Defining new attribute" + paramName + " with value " + paramValue);
						DMLConfig.setTextValue(this.xml_root, paramName, paramValue);
					}
					
				}
			} // end if (otherConfigNodeList != null && otherConfigNodeList.getLength() > 0){
		} catch (Exception e){
			LOG.error("Failed in merge default config file with optional config file",e);
			throw new ParseException("ERROR: error merging config file" + otherConfig.config_file_name + " with " + config_file_name);
		}
	}
	
	/**
	 * Method to parse configuration
	 * @throws ParserConfigurationException
	 * @throws SAXException
	 * @throws IOException
	 */
	private void parseConfig () throws ParserConfigurationException, SAXException, IOException 
	{
		DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
		factory.setIgnoringComments(true); //ignore XML comments
		DocumentBuilder builder = factory.newDocumentBuilder();
		Document domTree = null;
		if (config_file_name.startsWith("hdfs:") ||
		    config_file_name.startsWith("gpfs:") )  // config file from DFS
		{
			if( !LocalFileUtils.validateExternalFilename(config_file_name, true) )
				throw new IOException("Invalid (non-trustworthy) hdfs config filename.");
			FileSystem DFS = FileSystem.get(ConfigurationManager.getCachedJobConf());
            Path configFilePath = new Path(config_file_name);
            domTree = builder.parse(DFS.open(configFilePath));  
		}
		else  // config from local file system
		{
			if( !LocalFileUtils.validateExternalFilename(config_file_name, false) )
				throw new IOException("Invalid (non-trustworthy) local config filename.");
			domTree = builder.parse(config_file_name);
		}
		
		xml_root = domTree.getDocumentElement();		
	}
	
	/**
	 * Method to get string value of a configuration parameter
	 * Handles processing of configuration parameters 
	 * @param tagName the name of the DMLConfig parameter being retrieved
	 * @return a string representation of the DMLConfig parameter value.  
	 */
	public String getTextValue(String tagName) 
	{
		//get the actual value
		String retVal = (xml_root!=null)?getTextValue(xml_root,tagName):null;
		
		if (retVal == null)
		{
			if( _defaultVals.containsKey(tagName) )
				retVal = _defaultVals.get(tagName);
			else
				LOG.error("Error: requested dml configuration property '"+tagName+"' is invalid.");
		}
		
		return retVal;
	}
	
	public int getIntValue( String tagName )
	{
		return Integer.parseInt( getTextValue(tagName) );
	}
	
	public boolean getBooleanValue( String tagName )
	{
		return Boolean.parseBoolean( getTextValue(tagName) );
	}
	
	
	/**
	 * Method to get the string value of an element identified by tag
	 * @param ele
	 * @param tagName
	 * @return
	 */
	private static String getTextValue(Element element, String tagName) {
		String textVal = null;
		NodeList list = element.getElementsByTagName(tagName);
		if (list != null && list.getLength() > 0) {
			Element elem = (Element) list.item(0);
			textVal = elem.getFirstChild().getNodeValue();
			
		}
		return textVal;
	}
	
	/**
	 * Method to update the string value of an element identified by tagname 
	 * @param ele
	 * @param tagName
	 * @param newTextValue
	 */
	private static void setTextValue(Element element, String tagName, String newTextValue) {
		
		NodeList list = element.getElementsByTagName(tagName);
		if (list != null && list.getLength() > 0) {
			Element elem = (Element) list.item(0);
			elem.getFirstChild().setNodeValue(newTextValue);	
		}
	}

	/**
	 * 
	 * @return
	 * @throws DMLRuntimeException
	 */
	public synchronized String serializeDMLConfig() 
		throws DMLRuntimeException
	{
		String ret = null;
		try
		{		
			Transformer transformer = TransformerFactory.newInstance().newTransformer();
			transformer.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "yes");
			//transformer.setOutputProperty(OutputKeys.INDENT, "yes");
			StreamResult result = new StreamResult(new StringWriter());
			DOMSource source = new DOMSource(xml_root);
			transformer.transform(source, result);
			ret = result.getWriter().toString();
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to serialize DML config.", ex);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param content
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static DMLConfig parseDMLConfig( String content ) 
		throws DMLRuntimeException
	{
		DMLConfig ret = null;
		try
		{
			//System.out.println(content);
			DocumentBuilder builder = DocumentBuilderFactory.newInstance().newDocumentBuilder();
			Document domTree = null;
			domTree = builder.parse( new ByteArrayInputStream(content.getBytes("utf-8")) );
			Element root = domTree.getDocumentElement();
			ret = new DMLConfig( root );
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to parse DML config.", ex);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 * @throws ParseException 
	 */
	public static DMLConfig readAndMergeConfigurationFiles( String optConfig ) 
		throws ParseException
	{
		// optional config specified overwrites/merge into the default config
		DMLConfig defaultConfig = null;
		DMLConfig optionalConfig = null;
		
		if( optConfig != null ) { // the optional config is specified
			try { // try to get the default config first 
				defaultConfig = new DMLConfig(DEFAULT_SYSTEMML_CONFIG_FILEPATH, true);
			} catch (Exception e) { // it is ok to not have the default
				defaultConfig = null;
				LOG.warn("Default config file " + DEFAULT_SYSTEMML_CONFIG_FILEPATH + " not provided ");
			}
			try { // try to get the optional config next
				optionalConfig = new DMLConfig(optConfig, false);	
			} 
			catch (ParseException e) { // it is not ok as the specification is wrong
				optionalConfig = null;
				throw e;
			}
			if (defaultConfig != null) {
				try {
					defaultConfig.merge(optionalConfig);
				}
				catch(ParseException e){
					defaultConfig = null;
					throw e;
				}
			}
			else {
				defaultConfig = optionalConfig;
			}
		}
		else { // the optional config is not specified
			try { // try to get the default config 
				defaultConfig = new DMLConfig(DEFAULT_SYSTEMML_CONFIG_FILEPATH, false);
			} catch (ParseException e) { // it is not OK to not have the default
				defaultConfig = null;
				throw e;
			}
		}
		
		return defaultConfig;
	}

	/**
	 * 
	 * @return
	 */
	public String getConfigInfo() 
	{
		String[] tmpConfig = new String[]{ LOCAL_TMP_DIR,SCRATCH_SPACE,OPTIMIZATION_LEVEL,
				                     NUM_REDUCERS, DEFAULT_BLOCK_SIZE,
				                     YARN_APPMASTER, YARN_APPMASTERMEM, YARN_MAPREDUCEMEM,
				                     NUM_MERGE_TASKS, NUM_SOW_THREADS,NUM_REAP_THREADS,
				                     SOWER_WAIT_INTERVAL,REAPER_WAIT_INTERVAL,NIMBLE_SCRATCH }; 
		
		StringBuilder sb = new StringBuilder();
		for( String tmp : tmpConfig )
		{
			sb.append("INFO: ");
			sb.append(tmp);
			sb.append(": ");
			sb.append(getTextValue(tmp));
			sb.append("\n");
		}
		
		return sb.toString();
	}
	
	/**
	 * 
	 * @param amMem
	 * @param mrMem
	 */
	public void updateYarnMemorySettings(String amMem, String mrMem)
	{
		//app master memory
		NodeList list1 = xml_root.getElementsByTagName(YARN_APPMASTERMEM);
		if (list1 != null && list1.getLength() > 0) {
			Element elem = (Element) list1.item(0);
			elem.getFirstChild().setNodeValue(String.valueOf(amMem));
		}
		
		//mapreduce memory
		NodeList list2 = xml_root.getElementsByTagName(YARN_MAPREDUCEMEM);
		if (list2 != null && list2.getLength() > 0) {
			Element elem = (Element) list2.item(0);
			elem.getFirstChild().setNodeValue(String.valueOf(mrMem));
		}
	}
	
	/**
	 * 
	 * @param key
	 * @return
	 */
	public static String getDefaultTextValue( String key )
	{
		return _defaultVals.get( key );
	}
	
}
