/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.Map.Entry;

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;
import javax.xml.stream.XMLStreamWriter;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.ExternalFunctionStatement;
import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.ExternalFunctionProgramBlockCP;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheException;
import com.ibm.bi.dml.runtime.controlprogram.caching.LazyWriteBuffer;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDHandler;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.CPInstructionParser;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.Data;
import com.ibm.bi.dml.runtime.instructions.cp.FunctionCallCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.RandCPInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.MRInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

/**
 * DML Instructions Performance Test Tool: 
 * 
 * Creates an offline performance profile (required once per installation) of DML instructions.
 * The profile is a combination of all individual statistical models trained per combination of 
 * instruction and test configuration. In order to train those models, we execute and measure
 * real executions of DML instructions on random input data. Finally, during runtime, the profile
 * is used by the costs estimator in order to create statistic estimates for cost-based optimization.
 * 
 * TODO perftesttool: create dir if not existing
 * TODO perftesttool: dml via text in code 
 * TODO clarify lapack usage in perftest tool
 * 
 * TODO gen input data via rand MR job instead of in-memory (otherwise we cannot work on large data)
 * 
 * TODO: complete all CP instructions
 * TODO: add support for MR instructions (beispiel fix MR_datasize=dim*10, sort-io exec)
 * TODO: add support for TestVariable.PARALLELISM 
 * TODO: add support for instruction-invariant cost functions
 * TODO: add support for constants such as IO throughput (e.g., DFSIOTest) 
 * TODO: add support for known external functions and DML scripts / functions 
 */
public class PerfTestTool 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//public parameters (used for estimation)
	public static final long    MIN_DATASIZE           = 1000;
	public static final long    MAX_DATASIZE           = 1000000; 
	public static final long    DEFAULT_DATASIZE       = 500000;//(MAX_DATASIZE-MIN_DATASIZE)/2;
	public static final long    DATASIZE_MR_SCALE      = 20;
	public static final double  MIN_DIMSIZE            = 1;
	public static final double  MAX_DIMSIZE            = 1000; 
	public static final double  MIN_SPARSITY           = 0.1;
	public static final double  MAX_SPARSITY           = 1.0;
	public static final double  DEFAULT_SPARSITY       = 0.5;//(MAX_SPARSITY-MIN_SPARSITY)/2;
	public static final double  MIN_SORT_IO_MEM        = 10;
	public static final double  MAX_SORT_IO_MEM        = 500;
	public static final double  DEFAULT_SORT_IO_MEM    = 256; //BI: default 256MB, hadoop: default 100MB
	
	//internal parameters
	private static final boolean READ_STATS_ON_STARTUP  = false;
	private static final int     TEST_REPETITIONS       = 10; 
	private static final int     NUM_SAMPLES_PER_TEST   = 11; 
	private static final int     MODEL_MAX_ORDER        = 2;
	private static final boolean MODEL_INTERCEPT        = true;
	
	private static final String  PERF_TOOL_DIR          = "./conf/PerfTestTool/";
	private static final String  PERF_RESULTS_FNAME     = PERF_TOOL_DIR + "%id%.dat";
	private static final String  PERF_PROFILE_FNAME     = PERF_TOOL_DIR + "performance_profile.xml";
	private static final String  DML_SCRIPT_FNAME       = "./src/com/ibm/bi/dml/runtime/controlprogram/parfor/opt/PerfTestToolRegression.dml";
	private static final String  DML_TMP_FNAME          = PERF_TOOL_DIR + "temp.dml";
	
	//XML profile tags and attributes
	private static final String  XML_PROFILE            = "profile";
	private static final String  XML_DATE               = "date";
	private static final String  XML_INSTRUCTION        = "instruction";
	private static final String  XML_ID                 = "id";
	private static final String  XML_NAME               = "name";
	private static final String  XML_COSTFUNCTION       = "cost_function";
	private static final String  XML_MEASURE            = "measure";
	private static final String  XML_VARIABLE           = "lvariable";
	private static final String  XML_INTERNAL_VARIABLES = "pvariables";
	private static final String  XML_DATAFORMAT         = "dataformat";
	private static final String  XML_ELEMENT_DELIMITER  = "\u002c"; //","; 
		
	//ID sequences for instructions and test definitions
	private static IDSequence _seqInst     = null;
	private static IDSequence _seqTestDef  = null;
	
	//registered instructions and test definitions
	private static HashMap<Integer, PerfTestDef>   _regTestDef        = null; 
	private static HashMap<Integer, Instruction>   _regInst           = null;
	private static HashMap<Integer, String>        _regInst_IDNames   = null;
	private static HashMap<String, Integer>        _regInst_NamesID   = null;
	private static HashMap<Integer, Integer[]>     _regInst_IDTestDef = null; 
	private static HashMap<Integer, Boolean>       _regInst_IDVectors = null;
	private static HashMap<Integer, IOSchema>      _regInst_IDIOSchema = null;
	
	protected static final Log LOG = LogFactory.getLog(PerfTestTool.class.getName());
	
	
	private static Integer[] _defaultConf  = null;
	private static Integer[] _MRConf  = null;
	
	//raw measurement data (instID, physical defID, results)
	private static HashMap<Integer,HashMap<Integer,LinkedList<Double>>> _results = null;
		
	//profile data 
	private static boolean    _flagReadData = false; 
	private static HashMap<Integer,HashMap<Integer,CostFunction>> _profile = null;
	
	public enum TestMeasure //logical test measure
	{
		EXEC_TIME,
		MEMORY_USAGE	
	}
	
	public enum TestVariable //logical test variable
	{
		DATA_SIZE,
		SPARSITY,
		PARALLELISM,
		
		//some mr specific conf properites
		SORT_IO_MEM
	}
	
	public enum InternalTestVariable //physical test variable
	{
		DATA_SIZE,
		DIM1_SIZE,
		DIM2_SIZE,
		DIM3_SIZE,
		SPARSITY,	
		NUM_THREADS,
		NUM_MAPPERS,
		NUM_REDUCERS,
		
		SORT_IO_MEM
	}
	
	public enum IOSchema
	{
		NONE_NONE,
		NONE_UNARY,
		UNARY_NONE,
		UNARY_UNARY,
		BINARY_NONE,
		BINARY_UNARY
	}
	
	public enum DataFormat //logical data format
	{
		DENSE,
		SPARSE
	}
	
	public enum TestConstants //logical test constants
	{
		DFS_READ_THROUGHPUT,
		DFS_WRITE_THROUGHPUT,
		LFS_READ_THROUGHPUT,
		LFS_WRITE_THROUGHPUT
	}
	
	static
	{
		//init repository
		_seqInst      = new IDSequence();
		_seqTestDef   = new IDSequence();		
		_regTestDef   = new HashMap<Integer, PerfTestDef>();
		_regInst      = new HashMap<Integer, Instruction>();
		_regInst_IDNames = new HashMap<Integer, String>();
		_regInst_NamesID = new HashMap<String, Integer>();		
		_regInst_IDTestDef = new HashMap<Integer, Integer[]>();
		_regInst_IDVectors = new HashMap<Integer, Boolean>();
		_regInst_IDIOSchema = new HashMap<Integer, IOSchema>();
		_results      = new HashMap<Integer, HashMap<Integer,LinkedList<Double>>>();
		_profile      = new HashMap<Integer, HashMap<Integer,CostFunction>>();
		_flagReadData = false;
		
		//load existing profile if required
		try
		{
			if( READ_STATS_ON_STARTUP )
				readProfile( PERF_PROFILE_FNAME );
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
	}

	/**
	 * 
	 * @throws DMLRuntimeException
	 */
	public static void lazyInit() 
		throws DMLRuntimeException
	{
		//read profile for first access
		if( !_flagReadData )
		{
			try
			{
				//register all testdefs and instructions
				registerTestConfigurations();
				registerInstructions();
				
				//read profile
				readProfile( PERF_PROFILE_FNAME );
			}
			catch(Exception ex)
			{
				throw new DMLRuntimeException(ex);
			}	
		}
		
		if( _profile == null )
			throw new DMLRuntimeException("Performance test results have not been loaded completely.");
	}

	/**
	 * 
	 * @param opStr
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static boolean isRegisteredInstruction(String opStr)
		throws DMLRuntimeException 
	{
		//init if required
		lazyInit();
		
		//determine if inst registered
		return _regInst_NamesID.containsKey(opStr);
	}
	
	/**
	 * 
	 * @param instName
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static CostFunction getCostFunction( String instName, TestMeasure measure, TestVariable variable, DataFormat dataformat )
		throws DMLRuntimeException
	{		
		//init if required
		lazyInit();
		
		CostFunction tmp = null;
		int instID = getInstructionID( instName );
		if( instID != -1 ) //existing profile
		{
			int tdefID = getMappedTestDefID(instID, measure, variable, dataformat);		
			tmp = _profile.get(instID).get(tdefID);
		}
		return tmp;
	}
	
	/**
	 * 
	 * @param measure
	 * @param variable
	 * @param dataformat
	 * @return
	 */
	public CostFunction getInvariantCostFunction( TestMeasure measure, TestVariable[] variable, DataFormat dataformat )
	{
		//TODO: implement for additional rewrites
		throw new RuntimeException("Not implemented yet.");
	}

	/**
	 * 
	 * @return
	 */
	@SuppressWarnings("all")
	public static boolean runTest()
	{
		boolean ret = false;
	
		try
		{
			Timing time = new Timing();
			time.start();
			
			//init caching
			LazyWriteBuffer.init();
			
			//register all testdefs and instructions
			registerTestConfigurations();
			registerInstructions();
			
			//execute tests for all confs and all instructions
			executeTest();
			
			//compute regression models
			int rows = NUM_SAMPLES_PER_TEST;
			int cols = MODEL_MAX_ORDER + (MODEL_INTERCEPT ? 1 : 0);
			HashMap<Integer,Long> tmp = writeResults( PERF_TOOL_DIR );
			computeRegressionModels( DML_SCRIPT_FNAME, DML_TMP_FNAME, PERF_TOOL_DIR, tmp.size(), rows, cols);
			readRegressionModels( PERF_TOOL_DIR, tmp);
			
			//execConstantRuntimeTest();
			//execConstantMemoryTest();
		
			//write final profile to XML file
			writeProfile(PERF_TOOL_DIR, PERF_PROFILE_FNAME);
			System.out.format("SystemML PERFORMANCE TEST TOOL: finished profiling (in %.2f min), profile written to "+PERF_PROFILE_FNAME+"%n", time.stop()/60000);
			
			ret = true;
		}
		catch(Exception ex)
		{
			LOG.error("Failed to run performance test.", ex);
		}
		
		return ret;
	}

	/**
	 * 
	 */
	private static void registerTestConfigurations()
	{
		//reset ID Sequence for consistent IDs
		_seqTestDef.reset();
		
		//register default testdefs //TODO
		TestMeasure[] M = new TestMeasure[]{ TestMeasure.EXEC_TIME/*, TestMeasure.MEMORY_USAGE*/ };
		DataFormat[] D =  new DataFormat[]{DataFormat.DENSE/*,DataFormat.SPARSE*/};
		Integer[] defaultConf = new Integer[M.length*D.length*2];		
		int i=0;
		for( TestMeasure m : M ) //for all measures
			for( DataFormat d : D ) //for all data formats
			{
				defaultConf[i++] = registerTestDef( new PerfTestDef(m, TestVariable.DATA_SIZE, d, InternalTestVariable.DATA_SIZE,
                        MIN_DATASIZE, MAX_DATASIZE, NUM_SAMPLES_PER_TEST ) );
				defaultConf[i++] = registerTestDef( new PerfTestDef(m, TestVariable.SPARSITY, d, InternalTestVariable.SPARSITY,
						MIN_SPARSITY, MAX_SPARSITY, NUM_SAMPLES_PER_TEST ) );
			}
		

		//register advanced (multi-dim) test defs
		//FIXME enable
		/*for( TestMeasure m : M ) //for all measures
			for( DataFormat d : D ) //for all data formats
			{
				registerTestDef( new PerfTestDef( m, TestVariable.DATA_SIZE, d,
                        new InternalTestVariable[]{InternalTestVariable.DIM1_SIZE,InternalTestVariable.DIM2_SIZE,InternalTestVariable.DIM3_SIZE}, 
                        MIN_DIMSIZE, MAX_DIMSIZE, NUM_SAMPLES_PER_TEST ) );
			}?*

			
		//register MR specific instructions FIXME: just for test
		/*Integer[] mrConf = new Integer[D.length];
		i = 0;
		for( DataFormat d : D )
		{
			mrConf[i++] = registerTestDef( new PerfTestDef(TestMeasure.EXEC_TIME, TestVariable.SORT_IO_MEM, d,
					                         InternalTestVariable.SORT_IO_MEM,
				                             MIN_SORT_IO_MEM, MAX_SORT_IO_MEM, NUM_SAMPLES_PER_TEST ) );
		}*/
		
		//set default testdefs
		_defaultConf = defaultConf;
		//_MRConf = mrConf;
	}
	
	/**
	 * 
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	private static void registerInstructions() 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//reset ID sequences for consistent IDs
		_seqInst.reset();
		
		///////
		// CP instructions
		
		//matrix multiply mmtsj
		registerInstruction( "CP"+Lop.OPERAND_DELIMITOR+"tsmm", CPInstructionParser.parseSingleInstruction("CP"+Lop.OPERAND_DELIMITOR+"tsmm"+Lop.OPERAND_DELIMITOR+"A"+Lop.DATATYPE_PREFIX+"MATRIX"+Lop.VALUETYPE_PREFIX+"DOUBLE"+Lop.OPERAND_DELIMITOR+"C"+Lop.DATATYPE_PREFIX+"MATRIX"+Lop.VALUETYPE_PREFIX+"DOUBLE"+Lop.OPERAND_DELIMITOR+MMTSJType.LEFT),
						     getDefaultTestDefs(), false, IOSchema.UNARY_UNARY ); 
		
		/*
		//matrix multiply 
		registerInstruction( "CP"+Lops.OPERAND_DELIMITOR+"ba+*", CPInstructionParser.parseSingleInstruction("CP"+Lops.OPERAND_DELIMITOR+"ba+*"+Lops.OPERAND_DELIMITOR+"A"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE"+Lops.OPERAND_DELIMITOR+"B"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE"+Lops.OPERAND_DELIMITOR+"C"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE"),
						     getDefaultTestDefs(), false, IOSchema.BINARY_UNARY ); 
		////registerInstruction( "CP"+Lops.OPERAND_DELIMITOR+"ba+*", CPInstructionParser.parseSingleInstruction("CP"+Lops.OPERAND_DELIMITOR+"ba+*"+Lops.OPERAND_DELIMITOR+"A"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE"+Lops.OPERAND_DELIMITOR+"B"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE"+Lops.OPERAND_DELIMITOR+"C"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE"),
		////		             changeToMuliDimTestDefs(TestVariable.DATA_SIZE, getDefaultTestDefs()) ); 
		//rand
		registerInstruction( "CP"+Lops.OPERAND_DELIMITOR+"Rand", CPInstructionParser.parseSingleInstruction("CP"+Lops.OPERAND_DELIMITOR+"Rand"+Lops.OPERAND_DELIMITOR+"rows=1"+Lops.OPERAND_DELIMITOR+"cols=1"+Lops.OPERAND_DELIMITOR+"rowsInBlock=1000"+Lops.OPERAND_DELIMITOR+"colsInBlock=1000"+Lops.OPERAND_DELIMITOR+"min=1.0"+Lops.OPERAND_DELIMITOR+"max=100.0"+Lops.OPERAND_DELIMITOR+"sparsity=1.0"+Lops.OPERAND_DELIMITOR+"seed=7"+Lops.OPERAND_DELIMITOR+"pdf=uniform"+Lops.OPERAND_DELIMITOR+"dir=."+Lops.OPERAND_DELIMITOR+"C"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE"),
				 			 getDefaultTestDefs(), false, IOSchema.NONE_UNARY );
		//matrix transpose
		registerInstruction( "CP"+Lops.OPERAND_DELIMITOR+"r'", CPInstructionParser.parseSingleInstruction("CP"+Lops.OPERAND_DELIMITOR+"r'"+Lops.OPERAND_DELIMITOR+"A"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE"+Lops.OPERAND_DELIMITOR+"C"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE"),
	 			 			 getDefaultTestDefs(), false, IOSchema.UNARY_UNARY );
		//sum
		registerInstruction( "CP"+Lops.OPERAND_DELIMITOR+"uak+", CPInstructionParser.parseSingleInstruction("CP"+Lops.OPERAND_DELIMITOR+"uak+"+Lops.OPERAND_DELIMITOR+"A"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE"+Lops.OPERAND_DELIMITOR+"B"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE"), //needs B instead of C
	 			             getDefaultTestDefs(), false, IOSchema.UNARY_UNARY );
		//external function
		registerInstruction( "CP"+Lops.OPERAND_DELIMITOR+"extfunct", CPInstructionParser.parseSingleInstruction("CP"+Lops.OPERAND_DELIMITOR+"extfunct"+Lops.OPERAND_DELIMITOR+DMLProgram.DEFAULT_NAMESPACE+""+Lops.OPERAND_DELIMITOR+"execPerfTestExtFunct"+Lops.OPERAND_DELIMITOR+"1"+Lops.OPERAND_DELIMITOR+"1"+Lops.OPERAND_DELIMITOR+"A"+Lops.OPERAND_DELIMITOR+"C"),
	                         getDefaultTestDefs(), false, IOSchema.UNARY_UNARY );		
		//central moment
		registerInstruction( "CP"+Lops.OPERAND_DELIMITOR+"cm", CPInstructionParser.parseSingleInstruction("CP"+Lops.OPERAND_DELIMITOR+"cm"+Lops.OPERAND_DELIMITOR+"A"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE"+Lops.OPERAND_DELIMITOR+"2"+Lops.DATATYPE_PREFIX+"SCALAR"+Lops.VALUETYPE_PREFIX+"INT"+Lops.OPERAND_DELIMITOR+"c"+Lops.DATATYPE_PREFIX+"SCALAR"+Lops.VALUETYPE_PREFIX+"DOUBLE"),
	            			 getDefaultTestDefs(), true, IOSchema.UNARY_NONE ); 
		//co-variance
		registerInstruction( "CP"+Lops.OPERAND_DELIMITOR+"cov", CPInstructionParser.parseSingleInstruction("CP"+Lops.OPERAND_DELIMITOR+"cov"+Lops.OPERAND_DELIMITOR+"A"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE"+Lops.OPERAND_DELIMITOR+"B"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE"+Lops.OPERAND_DELIMITOR+"c"+Lops.DATATYPE_PREFIX+"SCALAR"+Lops.VALUETYPE_PREFIX+"DOUBLE"),
     						 getDefaultTestDefs(), true, IOSchema.BINARY_NONE );
		*/
		
		/*
		///////
		// MR instructions
		registerInstruction( "jobtypeMMRJ", createMRJobInstruction(JobType.MMRJ,
							                    MRInstructionParser.parseSingleInstruction("MR"+Lops.OPERAND_DELIMITOR+
							                    		                                   "rmm"+Lops.OPERAND_DELIMITOR+
							                    		                                   "0"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE"+Lops.OPERAND_DELIMITOR+
							                    		                                   "1"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE"+Lops.OPERAND_DELIMITOR+
							                    		                                   "2"+Lops.DATATYPE_PREFIX+"MATRIX"+Lops.VALUETYPE_PREFIX+"DOUBLE ")),
							 _MRConf, false, IOSchema.BINARY_UNARY ); 		

		*/
		/*ADD ADDITIONAL INSTRUCTIONS HERE*/
		
		
		
		//extend list to all (expensive) instructions; maybe also: createvar, assignvar, cpvar, rm, mv, setfilename, rmfilevar
		
	}
	
	private static Instruction createMRJobInstruction(JobType type, MRInstruction inst) 
	{
		MRJobInstruction mrinst = new MRJobInstruction(type);
		
		if( type == JobType.MMRJ )
		{
			ArrayList<String> inLab = new ArrayList<String>();
			ArrayList<String> outLab = new ArrayList<String>();
			inLab.add("A");
			inLab.add("B");
			outLab.add("C");
			
			mrinst.setMMRJInstructions(new String[]{"A","B"}, 
									   "", 
									   inst.toString(), 
									   "", 
									   "", 
									   new String[]{"C"},
									   new byte[]{2},
									   10, 1 );
			
		}
		
		
		return mrinst;
	}

	/**
	 * 
	 * @param def
	 * @return
	 */
	private static int registerTestDef( PerfTestDef def )
	{
		int ID = (int)_seqTestDef.getNextID();
		
		_regTestDef.put( ID, def );
		
		return ID;
	}
	
	/**
	 * 
	 * @param iname
	 * @param inst
	 * @param testDefIDs
	 * @param vectors
	 * @param schema
	 */
	private static void registerInstruction( String iname, Instruction inst, Integer[] testDefIDs, boolean vectors, IOSchema schema )
	{
		int ID = (int)_seqInst.getNextID();
		registerInstruction(ID, iname, inst, testDefIDs, vectors, schema);
	}
	
	/**
	 * 
	 * @param ID
	 * @param iname
	 * @param inst
	 * @param testDefIDs
	 * @param vector
	 * @param schema
	 */
	private static void registerInstruction( int ID, String iname, Instruction inst, Integer[] testDefIDs, boolean vector, IOSchema schema )
	{
		_regInst.put( ID, inst );
		_regInst_IDNames.put( ID, iname );
		_regInst_NamesID.put( iname, ID );
		_regInst_IDTestDef.put( ID, testDefIDs );
		_regInst_IDVectors.put( ID, vector );
		_regInst_IDIOSchema.put( ID, schema );
	}

	/**
	 * 
	 * @param instID
	 * @param measure
	 * @param variable
	 * @param dataformat
	 * @return
	 */
	private static int getMappedTestDefID( int instID, TestMeasure measure, TestVariable variable, DataFormat dataformat )
	{
		int ret = -1;
		
		for( Integer defID : _regInst_IDTestDef.get(instID) )
		{
			PerfTestDef def = _regTestDef.get(defID);
			if(   def.getMeasure()==measure 
				&& def.getVariable()==variable 
				&& def.getDataformat()==dataformat )
			{
				ret = defID;
				break;
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param measure
	 * @param lvariable
	 * @param dataformat
	 * @param pvariable
	 * @return
	 */
	@SuppressWarnings("unused")
	private static int getTestDefID( TestMeasure measure, TestVariable lvariable, DataFormat dataformat, InternalTestVariable pvariable )
	{
		return getTestDefID(measure, lvariable, dataformat, new InternalTestVariable[]{pvariable});
	}
	
	/**
	 * 
	 * @param measure
	 * @param lvariable
	 * @param dataformat
	 * @param pvariables
	 * @return
	 */
	private static int getTestDefID( TestMeasure measure, TestVariable lvariable, DataFormat dataformat, InternalTestVariable[] pvariables )
	{
		int ret = -1;
		
		for( Entry<Integer,PerfTestDef> e : _regTestDef.entrySet() )
		{
			PerfTestDef def = e.getValue();
			TestMeasure tmp1 = def.getMeasure();
			TestVariable tmp2 = def.getVariable();
			DataFormat tmp3 = def.getDataformat();
			InternalTestVariable[] tmp4 = def.getInternalVariables();
			
			if( tmp1==measure && tmp2==lvariable && tmp3==dataformat )
			{
				boolean flag = true;
				for( int i=0; i<tmp4.length; i++ )
					flag &= ( tmp4[i] == pvariables[i] );	
				
				if( flag )
				{
					ret = e.getKey();
					break;
				}
			}
		}

		return ret;
	}

	/**
	 * 
	 * @param instName
	 * @return
	 */
	private static int getInstructionID( String instName )
	{
		Integer ret = _regInst_NamesID.get( instName );
		return ( ret!=null )? ret : -1;
	}

	/**
	 * 
	 * @return
	 */
	@SuppressWarnings("unused")
	private static Integer[] getAllTestDefs()
	{
		return _regTestDef.keySet().toArray(new Integer[0]);
	}
	
	/**
	 * 
	 * @return
	 */
	private static Integer[] getDefaultTestDefs()
	{
		return _defaultConf;
	}
	
	/**
	 * 
	 * @param v
	 * @param IDs
	 * @return
	 */
	@SuppressWarnings("unused")
	private static Integer[] changeToMuliDimTestDefs( TestVariable v, Integer[] IDs )
	{
		Integer[] tmp = new Integer[IDs.length];
		
		for( int i=0; i<tmp.length; i++ )
		{
			PerfTestDef def = _regTestDef.get(IDs[i]);
			if( def.getVariable() == v ) //filter logical variables
			{
				//find multidim version
				InternalTestVariable[] in = null;
				switch( v )
				{
					case DATA_SIZE: 
						in = new InternalTestVariable[]{InternalTestVariable.DIM1_SIZE,InternalTestVariable.DIM2_SIZE,InternalTestVariable.DIM3_SIZE}; 
						break;
				}
				
				int newid = getTestDefID(def.getMeasure(), def.getVariable(), def.getDataformat(), in );
				
				//exchange testdef ID
				tmp[i] = newid;
			}
			else
			{
				tmp[i] = IDs[i];
			}
		}
		
		return tmp;
	}
	
	/**
	 * 
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	private static void executeTest( ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException, IOException
	{
		System.out.println("SystemML PERFORMANCE TEST TOOL:");
		
		//foreach registered instruction	
		for( Entry<Integer,Instruction> inst : _regInst.entrySet() )
		{
			int instID = inst.getKey();
			System.out.println( "Running INSTRUCTION "+_regInst_IDNames.get(instID) );
		
			Integer[] testDefIDs = _regInst_IDTestDef.get(instID);
			boolean vectors = _regInst_IDVectors.get(instID);
			IOSchema schema = _regInst_IDIOSchema.get(instID);
			
			//create tmp program block and set instruction
			Program prog = new Program();
			ProgramBlock pb = new ProgramBlock( prog );
			ArrayList<Instruction> ainst = new ArrayList<Instruction>();
			ainst.add( inst.getValue() );
			pb.setInstructions(ainst);
			
			ExecutionContext ec = new ExecutionContext();
			
			//foreach registered test configuration
			for( Integer defID : testDefIDs )
			{
				PerfTestDef def = _regTestDef.get(defID);
				TestMeasure m = def.getMeasure();
				TestVariable lv = def.getVariable();
				DataFormat df = def.getDataformat();
				InternalTestVariable[] pv = def.getInternalVariables();
				double min = def.getMin();
				double max = def.getMax();
				double samples = def.getNumSamples();
				
				System.out.println( "Running TESTDEF(measure="+m+", variable="+String.valueOf(lv)+" "+pv.length+", format="+String.valueOf(df)+")" );
				
				//vary input variable
				LinkedList<Double> dmeasure = new LinkedList<Double>();
				LinkedList<Double> dvariable = generateSequence(min, max, samples);					
				int plen = pv.length;
				
				if( plen == 1 ) //1D function 
				{
					for( Double var : dvariable )
					{
						dmeasure.add(executeTestCase1D(m, pv[0], df, var, pb, vectors, schema, ec));
					}
				}
				else //multi-dim function
				{
					//init index stack
					int[] index = new int[plen];
					for( int i=0; i<plen; i++ )
						index[i] = 0;
					
					//execute test 
					int dlen = dvariable.size();
					double[] buff = new double[plen];
					while( index[0]<dlen )
					{
						//set buffer values
						for( int i=0; i<plen; i++ )
							buff[i] = dvariable.get(index[i]);
						
						//core execution
						dmeasure.add(executeTestCaseMD(m, pv, df, buff, pb, schema, ec)); //not applicable for vector flag
						
						//increment indexes
						for( int i=plen-1; i>=0; i-- )
						{
							if(i==plen-1)
								index[i]++;
							else if( index[i+1] >= dlen )
							{
								index[i]++;
								index[i+1]=0;
							}
						}
					}
				}
				
								
				//append values to results
				if( !_results.containsKey(instID) )
					_results.put(instID, new HashMap<Integer, LinkedList<Double>>());
				_results.get(instID).put(defID, dmeasure);
	
			}
		}
	}
	
	/**
	 * 
	 * @param m
	 * @param v
	 * @param df
	 * @param varValue
	 * @param pb
	 * @param vectors
	 * @param schema
	 * 
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	private static double executeTestCase1D( TestMeasure m, InternalTestVariable v, DataFormat df, double varValue, ProgramBlock pb, boolean vectors, IOSchema schema, ExecutionContext ec ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException, IOException
	{
		double datasize = -1;
		double dim1 = -1, dim2 = -1;
		double sparsity = -1;
		//double sortio = -1;
		
		System.out.println( "VAR VALUE "+varValue );
	
		//set test variables
		switch ( v )
		{
			case DATA_SIZE:
				datasize = varValue;
				sparsity = DEFAULT_SPARSITY;
				break;
			case SPARSITY:
				datasize = DEFAULT_DATASIZE;
				sparsity = varValue;
				break;
			case SORT_IO_MEM: //FIXME
				datasize = DEFAULT_DATASIZE * DATASIZE_MR_SCALE;
				sparsity = DEFAULT_SPARSITY;
				//sortio = varValue;
				break;	
		}
		
		//set specific dimensions
		if( vectors )
		{
			dim1 = datasize;
			dim2 = 1;
		}
		else
		{
			dim1 = Math.sqrt( datasize );
			dim2 = dim1;
		}
		
		//instruction-specific configurations
		Instruction inst = pb.getInstruction(0); //always exactly one instruction
		if( inst instanceof RandCPInstruction )
		{
			RandCPInstruction rand = (RandCPInstruction) inst;
			rand.setRows((long)dim1);
			rand.setCols((long)dim2);
			rand.setSparsity(sparsity);
		}
		else if ( inst instanceof FunctionCallCPInstruction ) //ExternalFunctionInvocationInstruction
		{
			Program prog = pb.getProgram();
			ArrayList<DataIdentifier> in = new ArrayList<DataIdentifier>();
			DataIdentifier dat1 = new DataIdentifier("A");
			dat1.setDataType(DataType.MATRIX);
			dat1.setValueType(ValueType.DOUBLE);
			in.add(dat1);
			ArrayList<DataIdentifier> out = new ArrayList<DataIdentifier>();
			DataIdentifier dat2 = new DataIdentifier("C");
			dat2.setDataType(DataType.MATRIX);
			dat2.setValueType(ValueType.DOUBLE);
			out.add(dat2);
			HashMap<String, String> params = new HashMap<String, String>();
			params.put(ExternalFunctionStatement.CLASS_NAME, PerfTestExtFunctCP.class.getName());			
			ExternalFunctionProgramBlockCP fpb = new ExternalFunctionProgramBlockCP(prog, in, out, params, PERF_TOOL_DIR);	
			prog.addFunctionProgramBlock(DMLProgram.DEFAULT_NAMESPACE, "execPerfTestExtFunct", fpb);
		}
		else if ( inst instanceof MRJobInstruction )
		{
			//FIXME hardcoded for test
			//MMRJMR.SORT_IO_MEM = sortio;
			RunMRJobs.flagLocalModeOpt=false; //always use cluster mode
		}
		
		//generate input and output matrices
		LocalVariableMap vars = ec.getVariables();
		vars.removeAll();
		double mem1 = PerfTestMemoryObserver.getUsedMemory();
		if( schema!=IOSchema.NONE_NONE && schema!=IOSchema.NONE_UNARY )
			vars.put("A", generateInputDataset(PERF_TOOL_DIR+"/A", dim1, dim2, sparsity, df));
		if( schema==IOSchema.BINARY_NONE || schema==IOSchema.BINARY_UNARY || schema==IOSchema.UNARY_UNARY )
			vars.put("B", generateInputDataset(PERF_TOOL_DIR+"/B", dim1, dim2, sparsity, df));
		if( schema==IOSchema.NONE_UNARY || schema==IOSchema.UNARY_UNARY || schema==IOSchema.BINARY_UNARY)
			vars.put("C", generateEmptyResult(PERF_TOOL_DIR+"/C", dim1, dim2, df));
		double mem2 = PerfTestMemoryObserver.getUsedMemory();
		
		//foreach repetition
		double value = 0;
		for( int i=0; i<TEST_REPETITIONS; i++ )
		{
			System.out.println("run "+i);
			value += executeGenericProgramBlock( m, pb, ec );
		}
		value/=TEST_REPETITIONS;
		
		//result correction and print result
		switch( m )
		{
			case EXEC_TIME: System.out.println("--- RESULT: "+value+" ms"); break;
			case MEMORY_USAGE: 
				//System.out.println("--- RESULT: "+value+" byte"); 
				if( (mem2-mem1) > 0 )
					value = value + mem2-mem1; //correction: input sizes added
				System.out.println("--- RESULT: "+value+" byte"); break;
			default: System.out.println("--- RESULT: "+value); break;
		}
		
		return value;
	}
	
	/**
	 * 
	 * @param m
	 * @param v
	 * @param df
	 * @param varValue
	 * @param pb
	 * @param schema
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	private static double executeTestCaseMD( TestMeasure m, InternalTestVariable[] v, DataFormat df, double[] varValue, ProgramBlock pb, IOSchema schema, ExecutionContext ec ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException, IOException
	{
		//double datasize = DEFAULT_DATASIZE;
		double sparsity = DEFAULT_SPARSITY;
		double dim1 = -1;
		double dim2 = -1;
		double dim3 = -1;

		
		for( int i=0; i<v.length; i++ )
		{
			System.out.println( "VAR VALUE "+varValue[i] );
				
			switch( v[i] )
			{
				case DIM1_SIZE: dim1=varValue[i]; break;
				case DIM2_SIZE: dim2=varValue[i]; break;
				case DIM3_SIZE: dim3=varValue[i]; break;
			}
		}
		
		//generate input and output matrices
		LocalVariableMap vars = ec.getVariables();
		vars.removeAll();
		double mem1 = PerfTestMemoryObserver.getUsedMemory();
		if( schema!=IOSchema.NONE_NONE && schema!=IOSchema.NONE_UNARY )
			 vars.put("A", generateInputDataset(PERF_TOOL_DIR+"/A", dim1, dim2, sparsity, df));
		if( schema==IOSchema.BINARY_NONE || schema==IOSchema.BINARY_UNARY || schema==IOSchema.UNARY_UNARY )
			 vars.put("B", generateInputDataset(PERF_TOOL_DIR+"/B", dim2, dim3, sparsity, df));
		if( schema==IOSchema.NONE_UNARY || schema==IOSchema.UNARY_UNARY || schema==IOSchema.BINARY_UNARY)
			vars.put("C", generateEmptyResult(PERF_TOOL_DIR+"/C", dim1, dim3, df));
		double mem2 = PerfTestMemoryObserver.getUsedMemory();
		
		//foreach repetition
		double value = 0;
		for( int i=0; i<TEST_REPETITIONS; i++ )
		{
			System.out.println("run "+i);
			value += executeGenericProgramBlock( m, pb, ec );
		}
		value/=TEST_REPETITIONS;
		
		//result correction and print result
		switch( m )
		{
			case EXEC_TIME: System.out.println("--- RESULT: "+value+" ms"); break;
			case MEMORY_USAGE: 
				//System.out.println("--- RESULT: "+value+" byte"); 
				if( (mem2-mem1) > 0 )
					value = value + mem2-mem1; //correction: input sizes added
				System.out.println("--- RESULT: "+value+" byte"); break;
			default: System.out.println("--- RESULT: "+value); break;
		}
		
		return value;
	}
	
	/**
	 * 
	 * @param measure
	 * @param pb
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public static double executeGenericProgramBlock( TestMeasure measure, ProgramBlock pb, ExecutionContext ec ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		double value = 0;
		try
		{
			switch( measure )
			{
			 	case EXEC_TIME: 
			 		Timing time = new Timing(); 
			 		time.start();
			 		pb.execute( ec );
			 		value = time.stop();
			 		break;
			 	case MEMORY_USAGE:
			 		PerfTestMemoryObserver mo = new PerfTestMemoryObserver();
			 		mo.measureStartMem();
			 		Thread t = new Thread(mo);
			 		t.start();
			 		pb.execute( ec );
			 		mo.setStopped();
			 		value = mo.getMaxMemConsumption();
			 		t.join();
			 		break;
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		//clear matrixes from cache
		for( String str : ec.getVariables().keySet() )
		{
			Data dat = ec.getVariable(str); 
			if( dat instanceof MatrixObject )
				((MatrixObject)dat).clearData();		
		}
		
		return value;
	}

	/**
	 * 
	 * @param min
	 * @param max
	 * @param num
	 * @return
	 */
	public static LinkedList<Double> generateSequence( double min, double max, double num )
	{
		LinkedList<Double> data = new LinkedList<Double>();
		double increment = (max-min)/(num-1);
		
		for( int i=0; i<num; i++ )
			data.add( Double.valueOf(min+i*increment) );
		
		return data;
	}
	
	/**
	 * 
	 * @param fname
	 * @param datasize
	 * @param sparsity
	 * @param df
	 * @return
	 * @throws IOException
	 * @throws CacheException 
	 */
	public static MatrixObject generateInputDataset(String fname, double datasize, double sparsity, DataFormat df) 
		throws IOException, CacheException
	{
		int dim = (int)Math.sqrt( datasize );
		
		//create random test data
		double[][] d = generateTestMatrix(dim, dim, 1, 100, sparsity, 7);
		
		//create matrix block
		MatrixBlock mb = null;
		switch( df ) 
		{
			case DENSE:
				mb = new MatrixBlock(dim,dim,false);
				break;
			case SPARSE:
				mb = new MatrixBlock(dim,dim,true, (int)(sparsity*dim*dim));
				break;
		}
		
		//insert data
		for(int i=0; i < dim; i++)
			for(int j=0; j < dim; j++)
				if( d[i][j]!=0 )
					mb.setValue(i, j, d[i][j]);	
		
		MapReduceTool.deleteFileIfExistOnHDFS(fname);

		MatrixCharacteristics mc = new MatrixCharacteristics(dim, dim, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
		MatrixFormatMetaData md = new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		MatrixObject mo = new MatrixObject(ValueType.DOUBLE,fname,md);
		mo.acquireModify(mb);
		mo.release();
		mo.exportData(); //write to HDFS
		
		return mo;
	}
	
	/**
	 * 
	 * @param fname
	 * @param dim1
	 * @param dim2
	 * @param sparsity
	 * @param df
	 * @return
	 * @throws IOException
	 * @throws CacheException
	 */
	public static MatrixObject generateInputDataset(String fname, double dim1, double dim2, double sparsity, DataFormat df) 
		throws IOException, CacheException
	{		
		int d1 = (int) dim1;
		int d2 = (int) dim2;
		
		System.out.println(d1+" "+d2);
		
		//create random test data
		double[][] d = generateTestMatrix(d1, d2, 1, 100, sparsity, 7);
		
		//create matrix block
		MatrixBlock mb = null;
		switch( df ) 
		{
			case DENSE:
				mb = new MatrixBlock(d1,d2,false);
				break;
			case SPARSE:
				mb = new MatrixBlock(d1,d2,true, (int)(sparsity*dim1*dim2));
				break;
		}
		
		//insert data
		for(int i=0; i < d1; i++)
			for(int j=0; j < d2; j++)
				if( d[i][j]!=0 )
					mb.setValue(i, j, d[i][j]);		
		
		MapReduceTool.deleteFileIfExistOnHDFS(fname);
		
		MatrixCharacteristics mc = new MatrixCharacteristics(d1, d2, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
		MatrixFormatMetaData md = new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		MatrixObject mo = new MatrixObject(ValueType.DOUBLE,fname,md);
		mo.acquireModify(mb);
		mo.release();
		mo.exportData(); //write to HDFS
		
		return mo;
	}
	
	/**
	 * 
	 * @param fname
	 * @param datasize
	 * @param df
	 * @return
	 * @throws IOException
	 * @throws CacheException
	 */
	public static MatrixObject generateEmptyResult(String fname, double datasize, DataFormat df ) 
		throws IOException, CacheException
	{
		int dim = (int)Math.sqrt( datasize );
		
		/*
		MatrixBlock mb = null;
		switch( df ) 
		{
			case DENSE:
				mb = new MatrixBlock(dim,dim,false);
				break;
			case SPARSE:
				mb = new MatrixBlock(dim,dim,true);
				break;
		}*/
		
		MatrixCharacteristics mc = new MatrixCharacteristics(dim, dim, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
		MatrixFormatMetaData md = new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		MatrixObject mo = new MatrixObject(ValueType.DOUBLE,fname,md);
		
		return mo;
	}
	
	/**
	 * 
	 * @param fname
	 * @param dim1
	 * @param dim2
	 * @param df
	 * @return
	 * @throws IOException
	 * @throws CacheException
	 */
	public static MatrixObject generateEmptyResult(String fname, double dim1, double dim2, DataFormat df ) 
		throws IOException, CacheException
	{
		int d1 = (int)dim1;
		int d2 = (int)dim2;
		
		/*
		MatrixBlock mb = null;
		switch( df ) 
		{
			case DENSE:
				mb = new MatrixBlock(dim,dim,false);
				break;
			case SPARSE:
				mb = new MatrixBlock(dim,dim,true);
				break;
		}*/
		
		MatrixCharacteristics mc = new MatrixCharacteristics(d1, d2, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
		MatrixFormatMetaData md = new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		MatrixObject mo = new MatrixObject(ValueType.DOUBLE,fname,md);
		
		return mo;
	}
	

	/**
	 * NOTE: This is a copy of TestUtils.generateTestMatrix, it was replicated in order to prevent
	 * dependency of SystemML.jar to our test package.
	 */
	public static double[][] generateTestMatrix(int rows, int cols, double min, double max, double sparsity, long seed) {
		double[][] matrix = new double[rows][cols];
		Random random;
		if (seed == -1)
			random = new Random(System.nanoTime());
		else
			random = new Random(seed);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				if (random.nextDouble() > sparsity)
					continue;
				matrix[i][j] = (random.nextDouble() * (max - min) + min);
			}
		}

		return matrix;
	}


	/**
	 * 
	 * @param fname
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 * @throws XMLStreamException
	 * @throws IOException
	 */
	public static void externalReadProfile( String fname ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException, XMLStreamException, IOException
	{
		//validate external name (security issue)
		if( !LocalFileUtils.validateExternalFilename(fname, false) )
			throw new DMLRuntimeException("Invalid (non-trustworthy) external profile filename.");
		
		//register internals and read external profile
		registerTestConfigurations();
		registerInstructions();
		readProfile( fname );
	}

	/**
	 * 
	 * @param dirname
	 * @return
	 * @throws IOException
	 * @throws DMLUnsupportedOperationException
	 */
	@SuppressWarnings("all")
	private static HashMap<Integer,Long> writeResults( String dirname ) 
		throws IOException, DMLUnsupportedOperationException 
	{
		HashMap<Integer,Long> map = new HashMap<Integer, Long>();
		int count = 1;
		int offset = (MODEL_INTERCEPT ? 1 : 0);
		int cols = MODEL_MAX_ORDER + offset;
		
		for( Entry<Integer,HashMap<Integer,LinkedList<Double>>> inst : _results.entrySet() )
		{
			int instID = inst.getKey();
			HashMap<Integer,LinkedList<Double>> instCF = inst.getValue();
			
			for( Entry<Integer,LinkedList<Double>> cfun : instCF.entrySet() )
			{
				int tDefID = cfun.getKey();
				long ID = IDHandler.concatIntIDsToLong(instID, tDefID);
				LinkedList<Double> dmeasure = cfun.getValue();
				
				PerfTestDef def = _regTestDef.get(tDefID);
				LinkedList<Double> dvariable = generateSequence(def.getMin(), def.getMax(), NUM_SAMPLES_PER_TEST);
				int dlen = dvariable.size();
				int plen = def.getInternalVariables().length;
				
				//write variable data set
				CSVWriter writer1 = new CSVWriter( new FileWriter( dirname+count+"_in1.csv" ),',', CSVWriter.NO_QUOTE_CHARACTER);						
				if( plen == 1 ) //one dimensional function
				{
					//write 1, x, x^2, x^3, ...
					String[] sbuff = new String[cols];
					for( Double val : dvariable )
		    		{
		    			for( int j=0; j<cols; j++ )
	    					sbuff[j] = String.valueOf( Math.pow(val, j+1-offset) );
					    writer1.writeNext(sbuff);
		    		}
				}
				else // multi-dimensional function
				{
					//write 1, x,y,z,x^2,y^2,z^2, xy, xz, yz, xyz
					
					String[] sbuff = new String[(int)Math.pow(2,plen)-1+plen+offset-1]; 
					//String[] sbuff = new String[plen+offset];
					if(offset==1)
						sbuff[0]="1";
					
					//init index stack
					int[] index = new int[plen];
					for( int i=0; i<plen; i++ )
						index[i] = 0;
					
					//execute test 
					double[] buff = new double[plen];
					while( index[0]<dlen )
					{
						//set buffer values
						for( int i=0; i<plen; i++ )
							buff[i] = dvariable.get(index[i]);
						
						//core writing
						for( int i=1; i<=plen; i++ )
						{
							if( i==1 )
							{
								for( int j=0; j<plen; j++ )
									sbuff[offset+j] = String.valueOf( buff[j] );
								for( int j=0; j<plen; j++ )
									sbuff[offset+plen+j] = String.valueOf( Math.pow(buff[j],2) );
							}
							else if( i==2 )
							{
								int ix=0;
								for( int j=0; j<plen-1; j++ )
									for( int k=j+1; k<plen; k++, ix++ )
										sbuff[offset+2*plen+ix] = String.valueOf( buff[j]*buff[k] );
							}
							else if( i==plen )
							{
								//double tmp=1;
								//for( int j=0; j<plen; j++ )
								//	tmp *= buff[j];
								//sbuff[offset+2*plen+plen*(plen-1)/2] = String.valueOf(tmp);
							}
							else
								throw new DMLUnsupportedOperationException("More than 3 dims currently not supported.");
								
						}
							
						//for( int i=0; i<plen; i++ )	
	    				//	sbuff[offset+i] = String.valueOf( buff[i] );
						
					    writer1.writeNext(sbuff);

						//increment indexes
						for( int i=plen-1; i>=0; i-- )
						{
							if(i==plen-1)
								index[i]++;
							else if( index[i+1] >= dlen )
							{
								index[i]++;
								index[i+1]=0;
							}
						}
					}
				}				
			    writer1.close();
				
			    
				//write measure data set
				CSVWriter writer2 = new CSVWriter( new FileWriter( dirname+count+"_in2.csv" ),',', CSVWriter.NO_QUOTE_CHARACTER);		
				String[] buff2 = new String[1];
				for( Double val : dmeasure )
				{
					buff2[0] = String.valueOf( val );
					writer2.writeNext(buff2);
				}
				writer2.close();
			
				map.put(count, ID);
				count++;
			}
		}
		
		return map;
	}
	
	/**
	 * 
	 * @param dmlname
	 * @param dmltmpname
	 * @param dir
	 * @param models
	 * @param rows
	 * @param cols
	 * @throws IOException
	 * @throws ParseException
	 * @throws DMLException
	 */
	private static void computeRegressionModels( String dmlname, String dmltmpname, String dir, int models, int rows, int cols ) 
		throws IOException, ParseException, DMLException
	{
		//clean scratch space 
		//AutomatedTestBase.cleanupScratchSpace();
		
		//read DML template
		StringBuilder buffer = new StringBuilder();
		BufferedReader br = new BufferedReader( new FileReader(new File( dmlname )) );
	
		try
		{
			String line = null;
			while( (line=br.readLine()) != null )
			{
				buffer.append(line);
				buffer.append("\n");
			}
		}
		finally
		{
			if( br != null )
				br.close();
		}
		
		//replace parameters
		String template = buffer.toString();
		template = template.replaceAll("%numModels%", String.valueOf(models));
		template = template.replaceAll("%numRows%", String.valueOf(rows));
		template = template.replaceAll("%numCols%", String.valueOf(cols));
		template = template.replaceAll("%indir%", String.valueOf(dir));
		
		// write temp DML file
		File fout = new File(dmltmpname);
		FileOutputStream fos = new FileOutputStream(fout);
		try {
			fos.write(template.getBytes());
		}
		finally
		{
			if( fos != null )
				fos.close();
		}
		
		// execute DML script
		DMLScript.main(new String[] { "-f", dmltmpname });
	}
	
	/**
	 * 
	 * @param dname
	 * @param IDMapping
	 * @throws IOException
	 */
	private static void readRegressionModels( String dname, HashMap<Integer,Long> IDMapping ) 
		throws IOException
	{
		for( Entry<Integer,Long> e : IDMapping.entrySet() )
		{
			int count = e.getKey();
			long ID = e.getValue();
			int instID = IDHandler.extractIntIDFromLong(ID, 1);
			int tDefID = IDHandler.extractIntIDFromLong(ID, 2);
			
			//read file and parse
			LinkedList<Double> params = new LinkedList<Double>();
			CSVReader reader1 = new CSVReader( new FileReader(dname+count+"_out.csv"), ',' );
			String[] nextline = null;
			while( (nextline = reader1.readNext()) != null )
			{
				params.add(Double.parseDouble(nextline[0]));
			}
			reader1.close();
			
			double[] dparams = new double[params.size()];
			int i=0;
			for( Double d : params )
			{
				dparams[i] = d;
				i++;
			}
			
			//create new cost function
			boolean multidim = _regTestDef.get(tDefID).getInternalVariables().length > 1;
			CostFunction cf = new CostFunction(dparams, multidim); 
			
			//append to profile
			if( !_profile.containsKey(instID) )
				_profile.put(instID, new HashMap<Integer, CostFunction>());
			_profile.get(instID).put(tDefID, cf);
		}
	}

	/**
	 * 
	 * @param vars
	 * @return
	 */
	private static String serializeTestVariables( InternalTestVariable[] vars )
	{
		StringBuilder sb = new StringBuilder();
		for( int i=0; i<vars.length; i++ )
		{
			if( i>0 )
				sb.append( XML_ELEMENT_DELIMITER );
			sb.append( String.valueOf(vars[i]) );
		}
		return sb.toString();
	}
	
	/**
	 * 
	 * @param vars
	 * @return
	 */
	private static InternalTestVariable[] parseTestVariables(String vars)
	{
		StringTokenizer st = new StringTokenizer(vars, XML_ELEMENT_DELIMITER);
		InternalTestVariable[] v = new InternalTestVariable[st.countTokens()];
		for( int i=0; i<v.length; i++ )
			v[i] = InternalTestVariable.valueOf(st.nextToken());
		return v;
	}
	
	/**
	 * 
	 * @param vals
	 * @return
	 */
	private static String serializeParams( double[] vals )
	{
		StringBuilder sb = new StringBuilder();
		for( int i=0; i<vals.length; i++ )
		{
			if( i>0 )
				sb.append( XML_ELEMENT_DELIMITER );
			sb.append( String.valueOf(vals[i]) );
		}
		return sb.toString();
	}
	
	/**
	 * 
	 * @param valStr
	 * @return
	 */
	private static double[] parseParams( String valStr )
	{
		StringTokenizer st = new StringTokenizer(valStr, XML_ELEMENT_DELIMITER);
		double[] params = new double[st.countTokens()];
		for( int i=0; i<params.length; i++ )
			params[i] = Double.parseDouble(st.nextToken());
		return params;
	}
	
	/**
	 * 
	 * @param fname
	 * @throws XMLStreamException
	 * @throws IOException
	 */
	private static void readProfile( String fname ) 
		throws XMLStreamException, IOException
	{
		//init profile map
		_profile = new HashMap<Integer, HashMap<Integer,CostFunction>>();
		
		//read existing profile
		FileInputStream fis = new FileInputStream( fname );

		//xml parsing
		XMLInputFactory xif = XMLInputFactory.newInstance();
		XMLStreamReader xsr = xif.createXMLStreamReader( fis );
		
		try
		{
			int e = xsr.nextTag(); // profile start
			
			while( true ) //read all instructions
			{
				e = xsr.nextTag(); // instruction start
				if( e == XMLStreamConstants.END_ELEMENT )
					break; //reached profile end tag
				
				//parse instruction
				int ID = Integer.parseInt( xsr.getAttributeValue(null, XML_ID) );
				//String name = xsr.getAttributeValue(null, XML_NAME).trim().replaceAll(" ", Lops.OPERAND_DELIMITOR);
				HashMap<Integer, CostFunction> tmp = new HashMap<Integer, CostFunction>();
				_profile.put( ID, tmp );
				
				while( true )
				{
					e = xsr.nextTag(); // cost function start
					if( e == XMLStreamConstants.END_ELEMENT )
						break; //reached instruction end tag
					
					//parse cost function
					TestMeasure m = TestMeasure.valueOf( xsr.getAttributeValue(null, XML_MEASURE) );
					TestVariable lv = TestVariable.valueOf( xsr.getAttributeValue(null, XML_VARIABLE) );
					InternalTestVariable[] pv = parseTestVariables( xsr.getAttributeValue(null, XML_INTERNAL_VARIABLES) );
					DataFormat df = DataFormat.valueOf( xsr.getAttributeValue(null, XML_DATAFORMAT) );
					int tDefID = getTestDefID(m, lv, df, pv);
					
					xsr.next(); //read characters
					double[] params = parseParams(xsr.getText());
					boolean multidim = _regTestDef.get(tDefID).getInternalVariables().length > 1;
					CostFunction cf = new CostFunction( params, multidim );
					tmp.put(tDefID, cf);
				
					xsr.nextTag(); // cost function end
					//System.out.println("added cost function");
				}
			}
			xsr.close();
		}
		finally
		{
			if( fis != null )
				fis.close();
		}
		
		//mark profile as successfully read
		_flagReadData = true;
	}
	
	/**
	 * StAX for efficient streaming XML writing.
	 * 
	 * @throws IOException
	 * @throws XMLStreamException 
	 */
	private static void writeProfile( String dname, String fname ) 
		throws IOException, XMLStreamException 
	{
		//create initial directory and file 
		File dir =  new File( dname );
		if( !dir.exists() )
			dir.mkdir();
		File f = new File( fname );
		f.createNewFile();
		FileOutputStream fos = new FileOutputStream( f );
		
		//create document
		XMLOutputFactory xof = XMLOutputFactory.newInstance();
		XMLStreamWriter xsw = xof.createXMLStreamWriter( fos );
		//TODO use an alternative way for intentation
		//xsw = new IndentingXMLStreamWriter( xsw ); //remove this line if no indenting required
		
		try
		{
			//write document content
			xsw.writeStartDocument();
			xsw.writeStartElement( XML_PROFILE );
			xsw.writeAttribute(XML_DATE, String.valueOf(new Date()) );
			
			//foreach instruction (boundle of cost functions)
			for( Entry<Integer,HashMap<Integer,CostFunction>> inst : _profile.entrySet() )
			{
				int instID = inst.getKey();
				String instName = _regInst_IDNames.get( instID );
						
				xsw.writeStartElement( XML_INSTRUCTION ); 
				xsw.writeAttribute(XML_ID, String.valueOf( instID ));
				xsw.writeAttribute(XML_NAME, instName.replaceAll(Lop.OPERAND_DELIMITOR, " "));
				
				//foreach testdef cost function
				for( Entry<Integer,CostFunction> cfun : inst.getValue().entrySet() )
				{
					int tdefID = cfun.getKey();
					PerfTestDef def = _regTestDef.get(tdefID);
					CostFunction cf = cfun.getValue();
					
					xsw.writeStartElement( XML_COSTFUNCTION );
					xsw.writeAttribute( XML_ID, String.valueOf( tdefID ));
					xsw.writeAttribute( XML_MEASURE, def.getMeasure().toString() );
					xsw.writeAttribute( XML_VARIABLE, def.getVariable().toString() );
					xsw.writeAttribute( XML_INTERNAL_VARIABLES, serializeTestVariables(def.getInternalVariables()) );
					xsw.writeAttribute( XML_DATAFORMAT, def.getDataformat().toString() );
					xsw.writeCharacters(serializeParams( cf.getParams() ));
					xsw.writeEndElement();// XML_COSTFUNCTION
				}
				
				xsw.writeEndElement(); //XML_INSTRUCTION
			}
			
			xsw.writeEndElement();//XML_PROFILE
			xsw.writeEndDocument();
			xsw.close();
		}
		finally
		{
			if( fos != null )
				fos.close();
		}
	}

	
	
	/**
	 * Main for invoking the actual performance test in order to produce profile.xml
	 * 
	 * @param args
	 */
	public static void main(String[] args)
	{
		//execute the local / remote performance test
		PerfTestTool.runTest(); 
	}


}
