package com.ibm.bi.dml.utils.configuration;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.StringWriter;
import java.util.HashMap;

import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.utils.DMLRuntimeException;

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
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;


public class DMLConfig 
{
	private static final Log LOG = LogFactory.getLog(DMLConfig.class.getName());
	
	// external names of configuration properties 
	// (single point of change for all internal refs)
	public static final String LOCAL_TMP_DIR        = "localtmpdir";
	public static final String SCRATCH_SPACE        = "scratch";
	public static final String OPTIMIZATION_LEVEL   = "optlevel";	
	public static final String NUM_REDUCERS         = "numreducers";
	public static final String DEFAULT_BLOCK_SIZE   = "defaultblocksize"; 	
	public static final String NUM_MERGE_TASKS      = "NumMergeTasks";
	public static final String NUM_SOW_THREADS      = "NumberOfSowThreads";
	public static final String NUM_REAP_THREADS     = "NumberOfReapThreads";
	public static final String SOWER_WAIT_INTERVAL  = "SowerWaitInterval";
	public static final String REAPER_WAIT_INTERVAL = "ReaperWaitInterval";
	public static final String NIMBLE_SCRATCH       = "NimbleScratch";

	//internal config
	public static final String DEFAULT_SHARED_DIR_PERMISSION = "777"; //for local fs and hdfs
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
		_defaultVals.put(DEFAULT_BLOCK_SIZE,   "1000" );
		_defaultVals.put(NUM_MERGE_TASKS,      "4" );
		_defaultVals.put(NUM_SOW_THREADS,      "1" );
		_defaultVals.put(NUM_REAP_THREADS,     "1" );
		_defaultVals.put(SOWER_WAIT_INTERVAL,  "1000" );
		_defaultVals.put(REAPER_WAIT_INTERVAL, "1000" );
		_defaultVals.put(NIMBLE_SCRATCH,       "nimbleoutput" );	
	}
	
	
	/**
	 * Constructor to setup a DML configuration
	 * @param fileName
	 * @throws ParserConfigurationException
	 * @throws SAXException
	 * @throws IOException
	 */

	public DMLConfig(String fileName) 
		throws ParseException
	{
		config_file_name = fileName;
		try {
			parseConfig();
		}
		catch (Exception e){
		    //log error, since signature of generated ParseException doesn't allow to pass it 
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
							System.out.println("INFO: updating " + paramName + " with value " + paramValue);
						else 
							System.out.println("INFO: defining new attribute" + paramName + " with value " + paramValue);
						DMLConfig.setTextValue(this.xml_root, paramName, paramValue);
					}
					
				}
			} // end if (otherConfigNodeList != null && otherConfigNodeList.getLength() > 0){
		} catch (Exception e){
			LOG.error("Failed in merge default config file with optional config file",e);
			new ParseException("ERROR: error merging config file" + otherConfig.config_file_name + " with " + config_file_name);
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
		DocumentBuilder builder = DocumentBuilderFactory.newInstance().newDocumentBuilder();
		Document domTree = null;
		if (config_file_name.startsWith("hdfs:")) { // config file from hdfs
			FileSystem hdfs = FileSystem.get(new Configuration());
            Path configFilePath = new Path(config_file_name);
            domTree = builder.parse(hdfs.open(configFilePath));
		}
		else { // config from local file system
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
		String retVal = getTextValue(xml_root,tagName);
		
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
	 */
	public String getConfigInfo() 
	{
		String[] tmpConfig = new String[]{ LOCAL_TMP_DIR,SCRATCH_SPACE,OPTIMIZATION_LEVEL,
				                     NUM_REDUCERS,DEFAULT_BLOCK_SIZE, NUM_MERGE_TASKS, 
				                     NUM_SOW_THREADS,NUM_REAP_THREADS,SOWER_WAIT_INTERVAL,
				                     REAPER_WAIT_INTERVAL,NIMBLE_SCRATCH }; 
		
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
	 * @param key
	 * @return
	 */
	public static String getDefaultTextValue( String key )
	{
		return _defaultVals.get( key );
	}
}
