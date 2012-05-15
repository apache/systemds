package com.ibm.bi.dml.utils.configuration;

import java.io.IOException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;


public class DMLConfig 
{

    public static final int DEFAULT_BLOCK_SIZE = 1000;
    public static final int DEFAULT_NUM_REDUCERS = 75;
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

	public DMLConfig(String fileName) throws ParserConfigurationException, SAXException, IOException 
	{
		config_file_name = fileName;
		parseConfig();
	}
	
	public void merge(DMLConfig otherConfig) throws ParserConfigurationException, SAXException, IOException 
	{
		if (otherConfig == null) 
			return;
	
		// for each element in otherConfig, either overwrite existing value OR add to defaultConfig
		NodeList otherConfigNodeList = otherConfig.xml_root.getChildNodes();
		if (otherConfigNodeList != null && otherConfigNodeList.getLength() > 0){
			for (int i=0; i<otherConfigNodeList.getLength(); i++){
				org.w3c.dom.Node optionalConfigNode = otherConfigNodeList.item(i);
				if (optionalConfigNode.getNodeType() == org.w3c.dom.Node.ELEMENT_NODE){
					
					// try to find optional config node in default config node
					String paramName = optionalConfigNode.getNodeName();
					String paramValue = ((Element)optionalConfigNode).getFirstChild().getNodeValue();
					if (this.xml_root.getElementsByTagName(paramName) != null){
						DMLConfig.setTextValue(this.xml_root, paramName, paramValue);
						System.out.println("INFO: updating " + paramName + " with value " + paramValue);
					}
					else {
						System.out.println("ERROR: attempting to define new attribute " + paramValue + " not defined in default config");
						throw new ParserConfigurationException();
					}
					
				}
			} // end for (int i=0; i<otherConfigNodeList.getLength(); i++){
		} // end if (otherConfigNodeList != null && otherConfigNodeList.getLength() > 0){
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
		Document domTree = builder.parse(config_file_name);
		xml_root = domTree.getDocumentElement();		
		
	}
	
	/**
	 * Method to get string value of a configuration parameter
	 * @param tagName
	 * @return
	 */
	public String getTextValue(String tagName)
	{
		return getTextValue(xml_root,tagName);
	}
	
	
	
	/**
	 * Method to get the string value of an element identified by tag
	 * @param ele
	 * @param tagName
	 * @return
	 */
	private static String getTextValue(Element ele, String tagName) {
		String textVal = null;
		NodeList nl = ele.getElementsByTagName(tagName);
		if (nl != null && nl.getLength() > 0) {
			Element el = (Element) nl.item(0);
			textVal = el.getFirstChild().getNodeValue();
			
		}
		return textVal;
	}
	
	/**
	 * Method to update the string value of an element identified by tagname 
	 * @param ele
	 * @param tagName
	 * @param newTextValue
	 */
	private static void setTextValue(Element ele, String tagName, String newTextValue) {
		
		NodeList nl = ele.getElementsByTagName(tagName);
		if (nl != null && nl.getLength() > 0) {
			Element el = (Element) nl.item(0);
			el.getFirstChild().setNodeValue(newTextValue);	
		}
	}
	
}
