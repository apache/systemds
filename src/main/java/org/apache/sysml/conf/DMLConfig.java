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

import java.io.ByteArrayInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.Map;

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
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.codegen.SpoofCompiler.CompilerType;
import org.apache.sysml.hops.codegen.SpoofCompiler.PlanSelector;
import org.apache.sysml.lops.Compression;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;


public class DMLConfig
{

	public static final String DEFAULT_SYSTEMML_CONFIG_FILEPATH = "./SystemML-config.xml";
	
	private static final Log LOG = LogFactory.getLog(DMLConfig.class.getName());
	
	// external names of configuration properties 
	// (single point of change for all internal refs)
	public static final String LOCAL_TMP_DIR        = "sysml.localtmpdir";
	public static final String SCRATCH_SPACE        = "sysml.scratch";
	public static final String OPTIMIZATION_LEVEL   = "sysml.optlevel";
	public static final String NUM_REDUCERS         = "sysml.numreducers";
	public static final String JVM_REUSE            = "sysml.jvmreuse";
	public static final String DEFAULT_BLOCK_SIZE   = "sysml.defaultblocksize";
	public static final String YARN_APPMASTER       = "sysml.yarn.appmaster";
	public static final String YARN_APPMASTERMEM    = "sysml.yarn.appmaster.mem";
	public static final String YARN_MAPREDUCEMEM    = "sysml.yarn.mapreduce.mem";
	public static final String YARN_APPQUEUE        = "sysml.yarn.app.queue"; 
	public static final String CP_PARALLEL_OPS      = "sysml.cp.parallel.ops";
	public static final String CP_PARALLEL_IO       = "sysml.cp.parallel.io";
	public static final String COMPRESSED_LINALG    = "sysml.compressed.linalg"; //auto, true, false
	public static final String NATIVE_BLAS          = "sysml.native.blas";
	public static final String CODEGEN              = "sysml.codegen.enabled"; //boolean
	public static final String CODEGEN_COMPILER     = "sysml.codegen.compiler"; //see SpoofCompiler.CompilerType
	public static final String CODEGEN_OPTIMIZER    = "sysml.codegen.optimizer"; //see SpoofCompiler.PlanSelector
	public static final String CODEGEN_PLANCACHE    = "sysml.codegen.plancache"; //boolean
	public static final String CODEGEN_LITERALS     = "sysml.codegen.literals"; //1..heuristic, 2..always
	
	public static final String EXTRA_FINEGRAINED_STATS = "sysml.stats.finegrained"; //boolean
	public static final String STATS_MAX_WRAP_LEN   = "sysml.stats.maxWrapLength"; //int
	public static final String EXTRA_GPU_STATS      = "sysml.stats.extraGPU"; //boolean
	public static final String EXTRA_DNN_STATS      = "sysml.stats.extraDNN"; //boolean
	public static final String AVAILABLE_GPUS       = "sysml.gpu.availableGPUs"; // String to specify which GPUs to use (a range, all GPUs, comma separated list or a specific GPU)
	public static final String SYNCHRONIZE_GPU      = "sysml.gpu.sync.postProcess"; // boolean: whether to synchronize GPUs after every instruction 
	public static final String EAGER_CUDA_FREE		= "sysml.gpu.eager.cudaFree"; // boolean: whether to perform eager CUDA free on rmvar
	// Fraction of available memory to use. The available memory is computer when the GPUContext is created
	// to handle the tradeoff on calling cudaMemGetInfo too often.
	public static final String GPU_MEMORY_UTILIZATION_FACTOR = "sysml.gpu.memory.util.factor";
	public static final String FLOATING_POINT_PRECISION = "sysml.floating.point.precision"; // String to specify the datatype to use internally: supported values are double, single

	// supported prefixes for custom map/reduce configurations
	public static final String PREFIX_MAPRED = "mapred";
	public static final String PREFIX_MAPREDUCE = "mapreduce";
	
	//internal config
	public static final String DEFAULT_SHARED_DIR_PERMISSION = "777"; //for local fs and DFS
	public static String LOCAL_MR_MODE_STAGING_DIR = null;
	
	//configuration default values
	private static HashMap<String, String> _defaultVals = null;

    private String _fileName = null;
	private Element _xmlRoot = null;
	private DocumentBuilder _documentBuilder = null;
	private Document _document = null;
	
	static
	{
		_defaultVals = new HashMap<>();
		_defaultVals.put(LOCAL_TMP_DIR,          "/tmp/systemml" );
		_defaultVals.put(SCRATCH_SPACE,          "scratch_space" );
		_defaultVals.put(OPTIMIZATION_LEVEL,     String.valueOf(OptimizerUtils.DEFAULT_OPTLEVEL.ordinal()) );
		_defaultVals.put(NUM_REDUCERS,           "10" );
		_defaultVals.put(JVM_REUSE,              "false" );
		_defaultVals.put(DEFAULT_BLOCK_SIZE,     String.valueOf(OptimizerUtils.DEFAULT_BLOCKSIZE) );
		_defaultVals.put(YARN_APPMASTER,         "false" );
		_defaultVals.put(YARN_APPMASTERMEM,      "2048" );
		_defaultVals.put(YARN_MAPREDUCEMEM,      "-1" );
		_defaultVals.put(YARN_APPQUEUE,    	     "default" );
		_defaultVals.put(CP_PARALLEL_OPS,        "true" );
		_defaultVals.put(CP_PARALLEL_IO,         "true" );
		_defaultVals.put(COMPRESSED_LINALG,      Compression.CompressConfig.AUTO.name() );
		_defaultVals.put(CODEGEN,                "false" );
		_defaultVals.put(CODEGEN_COMPILER,       CompilerType.AUTO.name() );
		_defaultVals.put(CODEGEN_OPTIMIZER,      PlanSelector.FUSE_COST_BASED_V2.name() );
		_defaultVals.put(CODEGEN_PLANCACHE,      "true" );
		_defaultVals.put(CODEGEN_LITERALS,       "1" );
		_defaultVals.put(NATIVE_BLAS,            "none" );
		_defaultVals.put(EXTRA_FINEGRAINED_STATS,"false" );
		_defaultVals.put(STATS_MAX_WRAP_LEN,     "30" );
		_defaultVals.put(EXTRA_GPU_STATS,        "false" );
		_defaultVals.put(EXTRA_DNN_STATS,        "false" );
		_defaultVals.put(GPU_MEMORY_UTILIZATION_FACTOR,      "0.9" );
		_defaultVals.put(AVAILABLE_GPUS,         "-1");
		_defaultVals.put(SYNCHRONIZE_GPU,        "true" );
		_defaultVals.put(EAGER_CUDA_FREE,        "false" );
		_defaultVals.put(FLOATING_POINT_PRECISION,        	 "double" );
	}
	
	public DMLConfig()
	{
		
	}

	public DMLConfig(String fileName) 
		throws ParseException, FileNotFoundException
	{
		this( fileName, false );
	}
	
	public DMLConfig(String fileName, boolean silent) 
		throws ParseException, FileNotFoundException
	{
		_fileName = fileName;
		try {
			parseConfig();
		} catch (FileNotFoundException fnfe) {
			LOCAL_MR_MODE_STAGING_DIR = getTextValue(LOCAL_TMP_DIR) + "/hadoop/mapred/staging";
			throw fnfe;
		} catch (Exception e){
		    //log error, since signature of generated ParseException doesn't allow to pass it 
			if( !silent )
				LOG.error("Failed to parse DML config file ",e);
			throw new ParseException("ERROR: error parsing DMLConfig file " + fileName);
		}
		
		LOCAL_MR_MODE_STAGING_DIR = getTextValue(LOCAL_TMP_DIR) + "/hadoop/mapred/staging";
	}
	
	public DMLConfig( Element root )
	{
		_xmlRoot = root;
	}

	/**
	 * Method to parse configuration
	 * @throws ParserConfigurationException
	 * @throws SAXException
	 * @throws IOException
	 */
	private void parseConfig () throws ParserConfigurationException, SAXException, IOException 
	{
		DocumentBuilder builder = getDocumentBuilder();
		_document = null;
		if( _fileName.startsWith("hdfs:") || _fileName.startsWith("gpfs:")
			|| IOUtilFunctions.isObjectStoreFileScheme(new Path(_fileName)) )
		{
			Path configFilePath = new Path(_fileName);
			FileSystem DFS = IOUtilFunctions.getFileSystem(configFilePath);
			_document = builder.parse(DFS.open(configFilePath));
		}
		else  // config from local file system
		{
			_document = builder.parse(_fileName);
		}

		_xmlRoot = _document.getDocumentElement();
	}

	private DocumentBuilder getDocumentBuilder() throws ParserConfigurationException {
		if (_documentBuilder == null) {
			DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
			factory.setIgnoringComments(true); //ignore XML comments
			_documentBuilder = factory.newDocumentBuilder();
		}
		return _documentBuilder;
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
		String retVal = (_xmlRoot!=null)?getTextValue(_xmlRoot,tagName):null;
		
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
	
	public double getDoubleValue( String tagName )
	{
		return Double.parseDouble( getTextValue(tagName) );
	}
	
	/**
	 * Method to get the string value of an element identified by a tag name
	 * @param element the DOM element
	 * @param tagName the tag name
	 * @return the string value of the element
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
	 * Method to update the key value
	 * @param paramName parameter name
	 * @param paramValue parameter value
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void setTextValue(String paramName, String paramValue) throws DMLRuntimeException {
		if(_xmlRoot != null) {
			NodeList list = _xmlRoot.getElementsByTagName(paramName);
			if (list != null && list.getLength() > 0) {
				Element elem = (Element) list.item(0);
				elem.getFirstChild().setNodeValue(paramValue);
			} else {
				Node value = _document.createTextNode(paramValue);
				Node element = _document.createElement(paramName);
				element.appendChild(value);
				_xmlRoot.appendChild(element);
			}
		} else {
			try {
				DocumentBuilder builder = getDocumentBuilder();
				String configString = "<root><" + paramName + ">"+paramValue+"</" + paramName + "></root>";
				_document = builder.parse(new ByteArrayInputStream(configString.getBytes("UTF-8")));
				_xmlRoot = _document.getDocumentElement();
			} catch (Exception e) {
				throw new DMLRuntimeException("Unable to set config value", e);
			}
		}
	}

	/**
	 * Get a map of key/value pairs of all configurations w/ the prefix 'mapred'
	 * or 'mapreduce'.
	 * 
	 * @return map of mapred and mapreduce key/value pairs
	 */
	public Map<String, String> getCustomMRConfig()
	{
		HashMap<String, String> ret = new HashMap<>();
	
		//check for non-existing config xml tree
		if( _xmlRoot == null )
			return ret;
		
		//get all mapred.* and mapreduce.* tag / value pairs		
		NodeList list = _xmlRoot.getElementsByTagName("*");
		for( int i=0; list!=null && i<list.getLength(); i++ ) {
			if( list.item(i) instanceof Element &&
				(  ((Element)list.item(i)).getNodeName().startsWith(PREFIX_MAPRED) 
				|| ((Element)list.item(i)).getNodeName().startsWith(PREFIX_MAPREDUCE)) )
			{
				Element elem = (Element) list.item(i);
				ret.put(elem.getNodeName(), 
						elem.getFirstChild().getNodeValue());
			}
		}
		
		return ret;
	}
	
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
			DOMSource source = new DOMSource(_xmlRoot);
			transformer.transform(source, result);
			ret = result.getWriter().toString();
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to serialize DML config.", ex);
		}
		
		return ret;
	}
	
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
	 * Start with the internal default settings, then merge in the
	 * settings from any specified configuration file, if available.
	 * If it is not explicitly given, then merge in settings from
	 * the default configuration file location, if available.
	 *
	 * @param configPath User-defined path of the configuration file.
	 * @return dml configuration
	 * @throws ParseException if ParseException occurs
	 * @throws FileNotFoundException if FileNotFoundException occurs
	 */
	public static DMLConfig readConfigurationFile(String configPath)
		throws ParseException, FileNotFoundException
	{
		// Always start with the internal defaults
		DMLConfig config = new DMLConfig();

		// Merge in any specified or default configs if available
		if (configPath != null) {
			// specified
			try {
				config = new DMLConfig(configPath, false);
			} catch (FileNotFoundException fnfe) {
				LOG.error("Custom config file " + configPath + " not found.");
				throw fnfe;
			} catch (ParseException e) {
				throw e;
			}
		} else {
			// default
			try {
				config = new DMLConfig(DEFAULT_SYSTEMML_CONFIG_FILEPATH, false);
			} catch (FileNotFoundException fnfe) {
				LOG.info("Using internal default configuration settings.  If you wish to " +
						 "customize any settings, please supply a `SystemML-config.xml` file.");
				config = new DMLConfig();
			} catch (ParseException e) {
				throw e;
			}
		}
		return config;
	}

	public String getConfigInfo() 
	{
		String[] tmpConfig = new String[] { 
				LOCAL_TMP_DIR,SCRATCH_SPACE,OPTIMIZATION_LEVEL,
				NUM_REDUCERS, DEFAULT_BLOCK_SIZE,
				YARN_APPMASTER, YARN_APPMASTERMEM, YARN_MAPREDUCEMEM, 
				CP_PARALLEL_OPS, CP_PARALLEL_IO, NATIVE_BLAS,
				COMPRESSED_LINALG, 
				CODEGEN, CODEGEN_COMPILER, CODEGEN_OPTIMIZER, CODEGEN_PLANCACHE, CODEGEN_LITERALS,
				EXTRA_GPU_STATS, EXTRA_DNN_STATS, EXTRA_FINEGRAINED_STATS, STATS_MAX_WRAP_LEN,
				AVAILABLE_GPUS, SYNCHRONIZE_GPU, EAGER_CUDA_FREE, FLOATING_POINT_PRECISION
		}; 
		
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
	
	public void updateYarnMemorySettings(String amMem, String mrMem)
	{
		//app master memory
		NodeList list1 = _xmlRoot.getElementsByTagName(YARN_APPMASTERMEM);
		if (list1 != null && list1.getLength() > 0) {
			Element elem = (Element) list1.item(0);
			elem.getFirstChild().setNodeValue(String.valueOf(amMem));
		}
		
		//mapreduce memory
		NodeList list2 = _xmlRoot.getElementsByTagName(YARN_MAPREDUCEMEM);
		if (list2 != null && list2.getLength() > 0) {
			Element elem = (Element) list2.item(0);
			elem.getFirstChild().setNodeValue(String.valueOf(mrMem));
		}
	}

	public static String getDefaultTextValue( String key ) {
		return _defaultVals.get( key );
	}
	
	@Override
	public DMLConfig clone() {
		DMLConfig conf = new DMLConfig();
		conf._fileName = _fileName;
		conf._xmlRoot = (Element) _xmlRoot.cloneNode(true);
		
		return conf;
	}
}
