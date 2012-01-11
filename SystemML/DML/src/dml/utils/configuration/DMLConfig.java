package dml.utils.configuration;

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
	String config_file_name;
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
	
}
