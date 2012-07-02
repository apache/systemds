package com.ibm.bi.dml.utils.configuration;

import java.io.IOException;
import com.ibm.bi.dml.parser.*;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;


public class DMLConfig 
{
	// external names of configuration properties 
	// (single point of change for all internal refs)
	public static final String SCRATCH_SPACE        = "scratch";
	public static final String NUM_REDUCERS         = "numreducers";
	public static final String DEF_BLOCK_SIZE       = "defaultblocksize"; //TODO remove ambiguity (many different places)
	public static final String NUM_MERGE_TASKS      = "NumMergeTasks";
	public static final String NUM_SOW_THREADS      = "NumberOfSowThreads";
	public static final String NUM_REAP_THREADS     = "NumberOfReapThreads";
	public static final String SOWER_WAIT_INTERVAL  = "SowerWaitInterval";
	public static final String REAPER_WAIT_INTERVAL = "ReaperWaitInterval";
	public static final String NIMBLE_SCRATCH       = "NimbleScratch";
	
	//internal default values
	public static final int DEFAULT_BLOCK_SIZE = 1000;
    public static final int DEFAULT_NUM_REDUCERS = 75;
    public static final String LOCAL_MR_MODE_STAGING_DIR="/tmp/hadoop/mapred/staging";
	String config_file_name;
	
	public String getConfig_file_name() {
		return config_file_name;
	}
	Element xml_root;
	
	/**
	 * Constructor to setup a DML configuration
	 * @param fileName
	 * @throws ParserConfigurationException
	 * @throws SAXException
	 * @throws IOException
	 */

	public DMLConfig(String fileName) throws ParseException
	{
		config_file_name = fileName;
		try {
			parseConfig();
		}
		catch (Exception e){
			throw new ParseException("ERROR: error parsing DMLConfig file " + fileName);
		}
	}
	
	public void merge(DMLConfig otherConfig) throws ParseException
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
			new ParseException("ERROR: error merging config file" + otherConfig.config_file_name + " with " + this.config_file_name);
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
	 * Handles processing of conguration parameters 
	 * @param tagName the name of the DMLConfig parameter being retrieved
	 * @return a string representation of the DMLConfig parameter value.  
	 */
	public String getTextValue(String tagName) throws ParseException
	{
		String retVal = getTextValue(xml_root,tagName);
		if (retVal == null){
			throw new ParseException("ERROR: could not find parameter " + tagName + " in DMLConfig" );
		}
		
		return retVal;
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
	
}
