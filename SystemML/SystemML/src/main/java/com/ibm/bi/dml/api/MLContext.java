/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.api;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.hops.rewrite.ProgramRewriter;
import com.ibm.bi.dml.hops.rewrite.RewriteRemovePersistentReadWrite;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.parser.antlr4.DMLParserWrapper;
import com.ibm.bi.dml.parser.python.PyDMLParserWrapper;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.FunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContextFactory;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.cp.VariableCPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.data.RDDObject;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.utils.Explain;

/**
 * This is initial mockup API for Spark integration. Typical usage is as follows:
 * scala> import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes
 * scala> import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock
 * scala> import com.ibm.bi.dml.api.MLContext
 * scala> val args = Array("V.mtx", "W.mtx",  "H.mtx",  "2000", "1500",  "50",  "1",  "WOut.mtx",  "HOut.mtx")
 * scala> val V = sc.sequenceFile[MatrixIndexes, MatrixBlock]("hdfs://curly.almaden.ibm.com:9000/user/biadmin/V.mtx")
 * scala> val W = sc.sequenceFile[MatrixIndexes, MatrixBlock]("hdfs://curly.almaden.ibm.com:9000/user/biadmin/W.mtx")
 * scala> val H = sc.sequenceFile[MatrixIndexes, MatrixBlock]("hdfs://curly.almaden.ibm.com:9000/user/biadmin/H.mtx")
 * scala> val ml = new MLContext(sc)
 * scala> val V1 = org.apache.spark.api.java.JavaPairRDD.fromJavaRDD(V.toJavaRDD())
 * scala> val W1 = org.apache.spark.api.java.JavaPairRDD.fromJavaRDD(W.toJavaRDD())
 * scala> val H1 = org.apache.spark.api.java.JavaPairRDD.fromJavaRDD(H.toJavaRDD())
 * scala> ml.registerInput("V", V1)
 * scala> ml.registerInput("W", W1)
 * scala> ml.registerInput("H", H1)
 * scala> ml.registerOutput("H")
 * scala> ml.registerOutput("W")
 * scala> val outputs = ml.execute("GNMF.dml", args) 
 */
public class MLContext {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	// ----------------------------------------------------
	// TODO: Remove static by passing these variables through DMLScript
	public static SparkContext _sc = null; // Read while creating SystemML's spark context
	// ----------------------------------------------------
	
	private ArrayList<String> _inVarnames = null;
	private ArrayList<String> _outVarnames = null;
	private LocalVariableMap _variables = null; // temporary symbol table
	boolean parsePyDML = false;
		
	private DMLConfig _conf = null;
//	private String tmpHDFSDir = null; 
	
	public MLContext(SparkContext sc) {
		MLContext._sc = sc;
		//create default configuration
		_conf = new DMLConfig();
		ConfigurationManager.setConfig(_conf);
		DataExpression.REJECT_READ_UNKNOWN_SIZE = false;
	}
	
	
	public void registerInput(String varName, JavaPairRDD<MatrixIndexes,MatrixBlock> rdd) throws DMLRuntimeException {
		if(_variables == null)
			_variables = new LocalVariableMap();
		if(_inVarnames == null)
			_inVarnames = new ArrayList<String>();
		// TODO: Take care of dimensions
		MatrixCharacteristics mc = new MatrixCharacteristics(-1, -1, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
		MatrixObject mo = new MatrixObject(ValueType.DOUBLE, null, new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
		mo.setRDDHandle(new RDDObject(rdd));
		_variables.put(varName, mo);
		_inVarnames.add(varName);
	}
	
	public void registerOutput(String varName) throws DMLRuntimeException {
		if(_outVarnames == null)
			_outVarnames = new ArrayList<String>();
		_outVarnames.add(varName);
	}
	
	
	private void cleanupRuntimeProgram( Program prog, String[] outputs)
	{
		Map<String, FunctionProgramBlock> funcMap = prog.getFunctionProgramBlocks();
		if( funcMap != null && !funcMap.isEmpty() )
		{
			for( Entry<String, FunctionProgramBlock> e : funcMap.entrySet() )
			{
				FunctionProgramBlock fpb = e.getValue();
				for( ProgramBlock pb : fpb.getChildBlocks() )
					rCleanupRuntimeProgram(pb, outputs);
			}
		}
		
		for( ProgramBlock pb : prog.getProgramBlocks() )
			rCleanupRuntimeProgram(pb, outputs);
	}
	
	private void rCleanupRuntimeProgram( ProgramBlock pb, String[] outputs )
	{
		if( pb instanceof WhileProgramBlock )
		{
			WhileProgramBlock wpb = (WhileProgramBlock)pb;
			for( ProgramBlock pbc : wpb.getChildBlocks() )
				rCleanupRuntimeProgram(pbc,outputs);
		}
		else if( pb instanceof IfProgramBlock )
		{
			IfProgramBlock ipb = (IfProgramBlock)pb;
			for( ProgramBlock pbc : ipb.getChildBlocksIfBody() )
				rCleanupRuntimeProgram(pbc,outputs);
			for( ProgramBlock pbc : ipb.getChildBlocksElseBody() )
				rCleanupRuntimeProgram(pbc,outputs);
		}
		else if( pb instanceof ForProgramBlock )
		{
			ForProgramBlock fpb = (ForProgramBlock)pb;
			for( ProgramBlock pbc : fpb.getChildBlocks() )
				rCleanupRuntimeProgram(pbc,outputs);
		}
		else
		{
			ArrayList<Instruction> tmp = pb.getInstructions();
			for( int i=0; i<tmp.size(); i++ )
			{
				Instruction linst = tmp.get(i);
				if( linst instanceof VariableCPInstruction && ((VariableCPInstruction)linst).isRemoveVariable() )
				{
					VariableCPInstruction varinst = (VariableCPInstruction) linst;
					for( String var : outputs )
						if( varinst.isRemoveVariable(var) )
						{
							tmp.remove(i);
							i--;
							break;
						}
				}
			}
		}
	}
	
//	private void createTmpHDFSDir() throws IOException {
//		// Uncomment after adding tmp file
////		if(tmpHDFSDir == null) {
////			Random rand = new Random();
////			tmpHDFSDir = "systemml_tmp_" + rand.nextLong() + "_" + rand.nextLong() + "_" + rand.nextLong();
////			FileSystem hdfs = null; 
////			hdfs = FileSystem.get(new Configuration());
////			hdfs.mkdirs(new Path(tmpHDFSDir));
////		}
//	}
//	
//	/**
//	 * @param localFilePath
//	 * @return hdfsFilePath
//	 * @throws IOException
//	 */
//	private String putFileIntoTmpHDFSDir(String localFilePath) throws IOException {
//		throw new IOException("Writing to HDFS not implemented");
//	}
//	
//	private void deleteTmpHDFSDir() throws IOException {
//		// Uncomment after adding tmp file
////		if(tmpHDFSDir != null) {
////			FileSystem hdfs = FileSystem.get(new Configuration());
////			if(hdfs.exists(new Path(tmpHDFSDir)))
////				hdfs.delete(new Path(tmpHDFSDir), true);
////			hdfs.close();
////		}
//	}
	
	
	/**
	 * Execute DML script by passing named arguments
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param namedArgs
	 * @throws IOException
	 * @throws DMLException
	 * @throws ParseException 
	 */
	public HashMap<String, JavaPairRDD<MatrixIndexes,MatrixBlock>> execute(String dmlScriptFilePath, HashMap<String, String> namedArgs) throws IOException, DMLException, ParseException {
		String [] args = new String[namedArgs.size()];
		int i = 0;
		for(Entry<String, String> entry : namedArgs.entrySet()) {
			args[i] = entry.getKey() + "=" + entry.getValue();
			i++;
		}
		return runDMLScript(dmlScriptFilePath, args, true);
	}
	
	public HashMap<String, JavaPairRDD<MatrixIndexes,MatrixBlock>> execute(String dmlScriptFilePath, HashMap<String, String> namedArgs, boolean parsePyDML) throws IOException, DMLException, ParseException {
		String [] args = new String[namedArgs.size()];
		this.parsePyDML = parsePyDML;
		int i = 0;
		for(Entry<String, String> entry : namedArgs.entrySet()) {
			args[i] = entry.getKey() + "=" + entry.getValue();
			i++;
		}
		return runDMLScript(dmlScriptFilePath, args, true);
	}
	
	/**
	 * Execute DML script by passing positional arguments
	 * @param dmlScriptFilePath
	 * @param args
	 * @throws IOException
	 * @throws DMLException
	 * @throws ParseException 
	 */
	public HashMap<String, JavaPairRDD<MatrixIndexes,MatrixBlock>> execute(String dmlScriptFilePath, String [] args) throws IOException, DMLException, ParseException {
		return runDMLScript(dmlScriptFilePath, args, false);
	}
	
	public HashMap<String, JavaPairRDD<MatrixIndexes,MatrixBlock>> execute(String dmlScriptFilePath, String [] args, boolean parsePyDML) throws IOException, DMLException, ParseException {
		this.parsePyDML = parsePyDML;
		return runDMLScript(dmlScriptFilePath, args, false);
	}
	
	/**
	 * Execute DML script without any arguments
	 * @param dmlScriptFilePath
	 * @throws IOException
	 * @throws DMLException
	 * @throws ParseException 
	 */
	public HashMap<String, JavaPairRDD<MatrixIndexes,MatrixBlock>> execute(String dmlScriptFilePath) throws IOException, DMLException, ParseException {
		return runDMLScript(dmlScriptFilePath, null, false);
	}
	
	public HashMap<String, JavaPairRDD<MatrixIndexes,MatrixBlock>> execute(String dmlScriptFilePath, boolean parsePyDML) throws IOException, DMLException, ParseException {
		this.parsePyDML = parsePyDML;
		return runDMLScript(dmlScriptFilePath, null, false);
	}
	
	private HashMap<String, JavaPairRDD<MatrixIndexes,MatrixBlock>> runDMLScript(String dmlScriptFilePath, String [] args, boolean isNamedArgument) throws IOException, DMLException, ParseException {
		HashMap<String, JavaPairRDD<MatrixIndexes,MatrixBlock>> retVal = null;
		
		String dmlScriptStr = DMLScript.readDMLScript("-f", dmlScriptFilePath);		
		DMLScript.rtplatform = RUNTIME_PLATFORM.SPARK;
		
		// TODO: Set config file path
		
		HashMap<String, String> argVals = DMLScript.createArgumentsMap(isNamedArgument, args);
		//simplified compilation chain
		Program rtprog = null;
		
		//parsing
		DMLProgram prog = null;
		if(parsePyDML) {
			PyDMLParserWrapper parser = new PyDMLParserWrapper();
			prog = parser.parse(null, dmlScriptStr, argVals);
		}
		else {
			DMLParserWrapper parser = new DMLParserWrapper();
			prog = parser.parse(null, dmlScriptStr, argVals);
		}
		
		//language validate
		DMLTranslator dmlt = new DMLTranslator(prog);
		dmlt.liveVariableAnalysis(prog);			
		dmlt.validateParseTree(prog);
		
		//hop construct/rewrite
		dmlt.constructHops(prog);
		dmlt.rewriteHopsDAG(prog);
		
		String[] inputs = null; String[] outputs = null;
		if(_inVarnames != null) {
			System.out.println(_inVarnames.getClass().getName());
			inputs = _inVarnames.toArray(new String[0]);
		}
		else {
			inputs = new String[0];
		}
		if(_outVarnames != null) {
			System.out.println(_outVarnames.getClass().getName());
			outputs = _outVarnames.toArray(new String[0]);
		}
		else {
			outputs = new String[0];
		}
		
		//rewrite persistent reads/writes
		if(_variables != null) {
			RewriteRemovePersistentReadWrite rewrite = new RewriteRemovePersistentReadWrite(inputs, outputs);
			ProgramRewriter rewriter2 = new ProgramRewriter(rewrite);
			rewriter2.rewriteProgramHopDAGs(prog);
		}
		
		//lop construct and runtime prog generation
		dmlt.constructLops(prog);
		rtprog = prog.getRuntimeProgram(_conf);
		
		//final cleanup runtime prog
		cleanupRuntimeProgram(rtprog, outputs);
		
		System.out.println(Explain.explain(rtprog));
		
		//create and populate execution context
		ExecutionContext ec = ExecutionContextFactory.createContext(rtprog);	
		ec.setVariables(_variables);
		
		//core execute runtime program	
		rtprog.execute( ec );  
		
		for( String ovar : _outVarnames ) {
			if( _variables.keySet().contains(ovar) ) {
				if(retVal == null) {
					retVal = new HashMap<String, JavaPairRDD<MatrixIndexes,MatrixBlock>>();
				}
				retVal.put(ovar, ((SparkExecutionContext) ec).getRDDHandleForVariable(ovar));
			}
			else {
				throw new DMLException("Error: The variable " + ovar + " is not available as output after the execution of the DMLScript.");
			}
		}
		
		
		return retVal;
	}
}
