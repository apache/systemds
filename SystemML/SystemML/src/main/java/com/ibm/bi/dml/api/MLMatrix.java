package com.ibm.bi.dml.api;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SQLContext.QueryExecution;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

import com.ibm.bi.dml.api.datasource.MLBlock;
import com.ibm.bi.dml.api.datasource.functions.GetMIMBFromRow;
import com.ibm.bi.dml.api.datasource.functions.GetMLBlock;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.LanguageException;
import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.json.java.JSONObject;

/**
 * This class serves three purposes:
 * 1. It allows SystemML to fit nicely in MLPipeline by reducing number of reblocks.
 * 2. It simplifies interaction with SystemML's datasource api allowing user to easily read and write matrices without worrying 
 * too much about format, metadata and type of underlying RDDs.
 * 3. It provides off-the-shelf library for Distributed Blocked Matrix and reduces learning curve for using SystemML.
 * However, it is important to know that it is easy to abuse this off-the-shelf library and think it as replacement
 * to writing DML, which it is not. It does not provide any optimization between calls. A simple example
 * of the optimization that is conveniently skipped is: (t(m) %*% m)).
 * Also, note that this library is not thread-safe. The operator precedence is not exactly same as DML (as the precedence is
 * enforced by scala compiler), so please use appropriate brackets to enforce precedence. 

 import com.ibm.bi.dml.api.{MLContext, MLMatrix}
 import org.apache.spark.sql.SaveMode
 val ml = new MLContext(sc, false, true)
 val mat1 = MLMatrix.load(sqlContext, ml, "V_small.mtx", "binary")
 val mat2 = MLMatrix.load(sqlContext, ml, "W_small.mtx", "binary")
 val result = mat1.transpose() %*% mat2
 result.save("Result_small.mtx", "binary")
 
 */
public class MLMatrix extends DataFrame {
	private static final long serialVersionUID = -7005940673916671165L;
	protected static final Log LOG = LogFactory.getLog(DMLScript.class.getName());
	
	protected MLContext ml = null;
	protected long rlen = -1;
	protected long clen = -1;
	protected int brlen = DMLTranslator.DMLBlockSize; 
	protected int bclen = DMLTranslator.DMLBlockSize;
	protected long nnz;
	
	protected MLMatrix(SQLContext sqlContext, LogicalPlan logicalPlan) {
		super(sqlContext, logicalPlan);
	}

	protected MLMatrix(SQLContext sqlContext, QueryExecution queryExecution) {
		super(sqlContext, queryExecution);
	}
	
	// Only used internally to set a new MLMatrix after one of matrix operations.
	// Not to be used externally. 
	protected MLMatrix(DataFrame df, MLContext ml, long rlen, long clen, int brlen, int bclen, long nnz) {
		super(df.sqlContext(), df.logicalPlan());
		this.ml = ml;
		this.rlen = rlen;
		this.clen = clen;
		this.brlen = brlen;
		this.bclen = bclen;
		this.nnz = nnz;
	}
	
	// TODO: Add additional load to provide sep, missing values, etc. for CSV
	public static MLMatrix load(SQLContext sqlContext, MLContext ml, String filePath, String format) throws LanguageException, DMLRuntimeException {
		// First read metadata file and get rows, cols, rows_in_block, cols_in_block and nnz
		long rlen = -1; long clen = -1; long nnz = -1;
		int brlen = DMLTranslator.DMLBlockSize;  int bclen = DMLTranslator.DMLBlockSize; 
		JSONObject jsonObject = readMetadataFile(filePath+".mtd");
		if(jsonObject != null) {
			// Metadata file present 
			for( Object obj : jsonObject.entrySet() ){
				@SuppressWarnings("unchecked")
				Entry<Object,Object> e = (Entry<Object, Object>) obj;
	    		String key = e.getKey().toString();
	    		String val = e.getValue().toString();
	    		if(key.compareTo("rows") == 0) {
	    			rlen = Long.parseLong(val);
	    		}
	    		else if(key.compareTo("cols") == 0) {
	    			clen = Long.parseLong(val);
	    		}
	    		else if(key.compareTo("rows_in_block") == 0) {
	    			brlen = Integer.parseInt(val);
	    		}
	    		else if(key.compareTo("cols_in_block") == 0) {
	    			bclen = Integer.parseInt(val);
	    		}
	    		else if(key.compareTo("nnz") == 0) {
	    			nnz = Long.parseLong(val);
	    		}
			}
		}
		else {
			// Metadata file no present -- only proceed if csv file
			if(format.compareTo("csv") != 0) {
				throw new DMLRuntimeException("Metadata information expected for format \"" + format +"\"."
						+ "Either provide a " + filePath + ".mtd file or use the overloaded load(sqlContext, mlCtx, file, format, numRows, numCols, numRowsPerBlock, numColsPerBlock, nnz) method");
			}
		}
		
		return load(sqlContext, ml, filePath, format, rlen, clen, brlen, bclen, nnz);
	}
	
	/**
	 * Load factory method where no metadata file reading is performed.
	 * 
	 * @param sqlContext
	 * @param ml
	 * @param filePath
	 * @param format
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param nnz
	 * @return
	 */
	public static MLMatrix load(SQLContext sqlContext, MLContext ml, String filePath, String format, long rlen, long clen, int brlen, int bclen, long nnz) {
		HashMap<String, String> parameters = new HashMap<String, String>();
		parameters.put("file", filePath);
		parameters.put("format", format);
		
		DataFrame df = sqlContext.load("com.ibm.bi.dml.api.datasource", parameters);
		MLMatrix retVal = new MLMatrix(df.sqlContext(), df.logicalPlan());
		retVal.ml = ml;
		retVal.rlen = rlen;
		retVal.clen = clen;
		retVal.brlen = brlen;
		retVal.bclen = bclen;
		retVal.nnz = nnz;
		return retVal;
	}
	
	/**
	 * Convenient method to save a MLMatrix.
	 */
	public void save(String filePath, String format) {
		HashMap<String, String> parameters = new HashMap<String, String>();
		parameters.put("file", filePath);
		parameters.put("format", format);
		this.save("com.ibm.bi.dml.api.datasource", SaveMode.Overwrite, parameters);
	}
	
	private static JSONObject readMetadataFile(String filename) throws LanguageException {
		
		JSONObject retVal = null;
		boolean exists = false;
		FileSystem fs = null;
		
		try {
			fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
		} catch (Exception e){
			LOG.error("ERROR: could not read the configuration file.");
			throw new LanguageException("ERROR: could not read the configuration file.", e);
		}
		
		Path pt = new Path(filename);
		try {
			if (fs.exists(pt)){
				exists = true;
			}
		} catch (Exception e){
			exists = false;
		}
	
		boolean isDirBoolean = false;
		try {
			if (exists && fs.getFileStatus(pt).isDirectory())
				isDirBoolean = true;
			else
				isDirBoolean = false;
		}
		catch(Exception e){
			LOG.error("ERROR: error validing whether path " + pt.toString() + " is directory or not");
        	throw new LanguageException("ERROR: error validing whether path " + pt.toString() + " is directory or not", e);			
		}
		
		// CASE: filename is a directory -- process as a directory
		if (exists && isDirBoolean){
			
			// read directory contents
			retVal = new JSONObject();
			
			FileStatus[] stats = null;
			
			try {
				stats = fs.listStatus(pt);
			}
			catch (Exception e){
				LOG.error(e.toString());
				LOG.error("ERROR: for MTD file in directory, error reading directory with MTD file " + pt.toString() + ": " + e.toString());
				throw new LanguageException("ERROR: for MTD file in directory, error reading directory with MTD file " + pt.toString() + ": " + e.toString());	
			}
			
			for(FileStatus stat : stats){
				Path childPath = stat.getPath(); // gives directory name
				if (childPath.getName().startsWith("part")){
					
					BufferedReader br = null;
					try {
						br = new BufferedReader(new InputStreamReader(fs.open(childPath)));
					}
					catch(Exception e){
						LOG.error(e.toString());
						LOG.error("ERROR: for MTD file in directory, error reading part of MTD file with path " + childPath.toString() + ": " + e.toString());
						throw new LanguageException("ERROR: for MTD file in directory, error reading part of MTD file with path " + childPath.toString() + e.toString());	
					}
					
					JSONObject childObj = null;
					try {
						childObj = JSONObject.parse(br);
					}
					catch(Exception e){
						LOG.error("ERROR: for MTD file in directory, error parsing part of MTD file with path " + childPath.toString() + ": " + e.toString());
						throw new LanguageException("ERROR: for MTD file in directory, error parsing part of MTD file with path " + childPath.toString() + ": " + e.toString());		
					}
					
			    	for( Object obj : childObj.entrySet() ){
						@SuppressWarnings("unchecked")
						Entry<Object,Object> e = (Entry<Object, Object>) obj;
			    		Object key = e.getKey();
			    		Object val = e.getValue();
			    		retVal.put(key, val);
					}
				}
			} // end for 
		}
		
		// CASE: filename points to a file
		else if (exists){
			
			BufferedReader br = null;
			
			// try reading MTD file
			try {
				br=new BufferedReader(new InputStreamReader(fs.open(pt)));
			} catch (Exception e){
				LOG.error("ERROR: reading MTD file with path " + pt.toString() + ": " + e.toString());
				throw new LanguageException("ERROR: reading with path " + pt.toString() + ": " + e.toString());
	        }
			
			// try parsing MTD file
			try {
				retVal =  JSONObject.parse(br);	
			} catch (Exception e){
				LOG.error("ERROR: parsing MTD file with path " + pt.toString() + ": " + e.toString());
				throw new LanguageException("ERROR: parsing MTD with path " + pt.toString() + ": " + e.toString());
	        }
		}
			
		return retVal;
	}
	
	private double getScalarBuiltinFunctionResult(String fn) throws IOException, DMLException, ParseException {
		if(fn.compareTo("nrow") == 0 || fn.compareTo("ncol") == 0) {
			if(ml == null) {
				throw new DMLRuntimeException("MLContext needs to be set");
			}
			ml.reset();
			ml.registerInput("left", getRDDLazily(this), rlen, clen, brlen, bclen);
			ml.registerOutput("output");
			String script = "left = read(\"ignore1.mtx\", rows=" + rlen + ", cols=" + clen + ", rows_in_block=" + brlen + ", cols_in_block=" + bclen + ", nnz=" + nnz + ", format=\"binary\");"
					+ "val = " + fn + "(left); "
					+ "output = matrix(val, rows=1, cols=1); "
					+ "write(output, \"ignore3.mtx\", rows_in_block=" + brlen + ", cols_in_block=" + bclen + ", format=\"binary\"); ";
			MLOutput out = ml.executeScript(script);
			List<Tuple2<MatrixIndexes, MatrixBlock>> result = out.getBinaryBlockedRDD("output").collect();
			if(result == null || result.size() != 1) {
				throw new DMLRuntimeException("Error while computing the function: " + fn);
			}
			return result.get(0)._2.getValue(0, 0);
		}
		else {
			throw new DMLRuntimeException("The function " + fn + " is not yet supported in MLMatrix");
		}
	}
	
	/**
	 * Gets or computes the number of rows.
	 * @return
	 * @throws ParseException 
	 * @throws DMLException 
	 * @throws IOException 
	 */
	public long numRows() throws IOException, DMLException, ParseException {
		if(clen != -1) {
			return clen;
		}
		else {
			return (long) getScalarBuiltinFunctionResult("nrow");
		}
	}
	
	/**
	 * Gets or computes the number of columns.
	 * @return
	 * @throws ParseException 
	 * @throws DMLException 
	 * @throws IOException 
	 */
	public long numCols() throws IOException, DMLException, ParseException {
		if(clen != -1) {
			return clen;
		}
		else {
			return (long) getScalarBuiltinFunctionResult("ncol");
		}
	}
	
	public int rowsPerBlock() {
		return brlen;
	}
	
	public int colsPerBlock() {
		return bclen;
	}
	
	private String getScript(String binaryOperator, long rightNumRows, long rightNumCols, long rightNNZ) {
		// Since blocksizes have already been checked, no need to pass them
		return 	"left = read(\"ignore1.mtx\", rows=" + rlen + ", cols=" + clen + ", rows_in_block=" + brlen + ", cols_in_block=" + bclen + ", nnz=" + nnz + ", format=\"binary\");"
				+ "right = read(\"ignore2.mtx\", rows=" + rightNumRows + ", cols=" + rightNumCols + ", rows_in_block=" + brlen + ", cols_in_block=" + bclen + ", nnz=" + rightNNZ + ", format=\"binary\");"
				+ "output = left " + binaryOperator + " right; "
				+ "write(output, \"ignore3.mtx\", rows_in_block=" + brlen + ", cols_in_block=" + bclen + ", format=\"binary\"); ";
	}
	
	private String getScalarBinaryScript(String binaryOperator, double scalar, boolean isScalarLeft) {
		if(isScalarLeft) {
			return 	"left = read(\"ignore1.mtx\", rows=" + rlen + ", cols=" + clen + ", format=\"binary\");"
					+ "output = " + scalar + " " + binaryOperator + " left ;"
					+ "write(output, \"ignore3.mtx\", rows_in_block=" + brlen + ", cols_in_block=" + bclen + ", format=\"binary\"); ";
		}
		else {
			return 	"left = read(\"ignore1.mtx\", rows=" + rlen + ", cols=" + clen + ", rows_in_block=" + brlen + ", cols_in_block=" + bclen + ", nnz=" + nnz + ", format=\"binary\");"
				+ "output = left " + binaryOperator + " " + scalar + ";"
				+ "write(output, \"ignore3.mtx\", rows_in_block=" + brlen + ", cols_in_block=" + bclen + ", format=\"binary\"); ";
		}
	}
	
	static JavaPairRDD<MatrixIndexes, MatrixBlock> getRDDLazily(MLMatrix mat) {
		return mat.rdd().toJavaRDD().mapToPair(new GetMIMBFromRow());
	}
	
	private MLMatrix matrixBinaryOp(MLMatrix that, String op) throws IOException, DMLException, ParseException {
		if(ml == null) {
			throw new DMLRuntimeException("MLContext needs to be set");
		}
		if(brlen != that.brlen || bclen != that.bclen) {
			throw new DMLRuntimeException("Incompatible block sizes:" + brlen + "!=" +  that.brlen + " || " + bclen + "!=" + that.bclen);
		}
		
		if(op.compareTo("%*%") == 0) {
			if(clen != that.rlen) {
				throw new DMLRuntimeException("Dimensions mismatch:" + clen + "!=" +  that.rlen);
			}
		}
		else {
			if(rlen != that.rlen || clen != that.clen) {
				throw new DMLRuntimeException("Dimensions mismatch:" + rlen + "!=" +  that.rlen + " || " + clen + "!=" + that.clen);
			}
		}
		
		ml.reset();
		ml.registerInput("left", getRDDLazily(this), rlen, clen, brlen, bclen);
		ml.registerInput("right", getRDDLazily(that), that.rlen, that.clen, that.brlen, that.bclen);
		ml.registerOutput("output");
		MLOutput out = ml.executeScript(getScript(op, that.rlen, that.clen, that.nnz));
		RDD<Row> rows = out.getBinaryBlockedRDD("output").map(new GetMLBlock()).rdd();
		StructType schema = MLBlock.getDefaultSchemaForBinaryBlock();
		long estimatedNNZ = -1; // TODO: Estimate number of non-zeros after matrix-matrix operation
		if(op.compareTo("%*%") == 0) {
			return new MLMatrix(this.sqlContext().createDataFrame(rows.toJavaRDD(), schema), ml, rlen, that.clen, brlen, bclen, estimatedNNZ);
		}
		else {
			return new MLMatrix(this.sqlContext().createDataFrame(rows.toJavaRDD(), schema), ml, rlen, clen, brlen, bclen, estimatedNNZ);
		}
	}
	
	private MLMatrix scalarBinaryOp(Double scalar, String op, boolean isScalarLeft) throws IOException, DMLException, ParseException {
		if(ml == null) {
			throw new DMLRuntimeException("MLContext needs to be set");
		}
		ml.reset();
		ml.registerInput("left", getRDDLazily(this), rlen, clen, brlen, bclen);
		ml.registerOutput("output");
		MLOutput out = ml.executeScript(getScalarBinaryScript(op, scalar, isScalarLeft));
		RDD<Row> rows = out.getBinaryBlockedRDD("output").map(new GetMLBlock()).rdd();
		StructType schema = MLBlock.getDefaultSchemaForBinaryBlock();
		long estimatedNNZ = -1; // TODO: Estimate number of non-zeros after matrix-scalar operation
		return new MLMatrix(this.sqlContext().createDataFrame(rows.toJavaRDD(), schema), ml, rlen, clen, brlen, bclen, estimatedNNZ);
	}
	
	// ---------------------------------------------------
	// Simple operator loading but doesnot utilize the optimizer
	
	public MLMatrix $greater(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, ">");
	}
	
	public MLMatrix $less(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "<");
	}
	
	public MLMatrix $greater$eq(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, ">=");
	}
	
	public MLMatrix $less$eq(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "<=");
	}
	
	public MLMatrix $eq$eq(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "==");
	}
	
	public MLMatrix $bang$eq(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "!=");
	}
	
	public MLMatrix $up(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "^");
	}
	
	public MLMatrix exp(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "^");
	}
	
	public MLMatrix $plus(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "+");
	}
	
	public MLMatrix add(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "+");
	}
	
	public MLMatrix $minus(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "-");
	}
	
	public MLMatrix minus(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "-");
	}
	
	public MLMatrix $times(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "*");
	}
	
	public MLMatrix elementWiseMultiply(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "*");
	}
	
	public MLMatrix $div(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "/");
	}
	
	public MLMatrix divide(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "/");
	}
	
	public MLMatrix $percent$div$percent(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "%/%");
	}
	
	public MLMatrix integerDivision(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "%/%");
	}
	
	public MLMatrix $percent$percent(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "%%");
	}
	
	public MLMatrix modulus(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "%%");
	}
	
	public MLMatrix $percent$times$percent(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "%*%");
	}
	
	public MLMatrix multiply(MLMatrix that) throws IOException, DMLException, ParseException {
		return matrixBinaryOp(that, "%*%");
	}
	
	public MLMatrix transpose() throws IOException, DMLException, ParseException {
		if(ml == null) {
			throw new DMLRuntimeException("MLContext needs to be set");
		}
		ml.reset();
		ml.registerInput("left", getRDDLazily(this), rlen, clen, brlen, bclen);
		ml.registerOutput("output");
		String script = "left = read(\"ignore1.mtx\", rows=" + rlen + ", cols=" + clen + ", rows_in_block=" + brlen + ", cols_in_block=" + bclen + ", nnz=" + nnz + ", format=\"binary\");"
				+ "output = t(left); "
				+ "write(output, \"ignore3.mtx\", rows_in_block=" + brlen + ", cols_in_block=" + bclen + ", format=\"binary\"); ";
		MLOutput out = ml.executeScript(script);
		RDD<Row> rows = out.getBinaryBlockedRDD("output").map(new GetMLBlock()).rdd();
		StructType schema = MLBlock.getDefaultSchemaForBinaryBlock();
		return new MLMatrix(this.sqlContext().createDataFrame(rows.toJavaRDD(), schema), ml, clen, rlen, brlen, bclen, nnz);
	}
	
	// TODO: For 'scalar op matrix' operations: Do implicit conversions 
	public MLMatrix $plus(Double scalar) throws IOException, DMLException, ParseException {
		return scalarBinaryOp(scalar, "+", false);
	}
	
	public MLMatrix add(Double scalar) throws IOException, DMLException, ParseException {
		return scalarBinaryOp(scalar, "+", false);
	}
	
	public MLMatrix $minus(Double scalar) throws IOException, DMLException, ParseException {
		return scalarBinaryOp(scalar, "-", false);
	}
	
	public MLMatrix minus(Double scalar) throws IOException, DMLException, ParseException {
		return scalarBinaryOp(scalar, "-", false);
	}
	
	public MLMatrix $times(Double scalar) throws IOException, DMLException, ParseException {
		return scalarBinaryOp(scalar, "*", false);
	}
	
	public MLMatrix elementWiseMultiply(Double scalar) throws IOException, DMLException, ParseException {
		return scalarBinaryOp(scalar, "*", false);
	}
	
	public MLMatrix $div(Double scalar) throws IOException, DMLException, ParseException {
		return scalarBinaryOp(scalar, "/", false);
	}
	
	public MLMatrix divide(Double scalar) throws IOException, DMLException, ParseException {
		return scalarBinaryOp(scalar, "/", false);
	}
	
	public MLMatrix $greater(Double scalar) throws IOException, DMLException, ParseException {
		return scalarBinaryOp(scalar, ">", false);
	}
	
	public MLMatrix $less(Double scalar) throws IOException, DMLException, ParseException {
		return scalarBinaryOp(scalar, "<", false);
	}
	
	public MLMatrix $greater$eq(Double scalar) throws IOException, DMLException, ParseException {
		return scalarBinaryOp(scalar, ">=", false);
	}
	
	public MLMatrix $less$eq(Double scalar) throws IOException, DMLException, ParseException {
		return scalarBinaryOp(scalar, "<=", false);
	}
	
	public MLMatrix $eq$eq(Double scalar) throws IOException, DMLException, ParseException {
		return scalarBinaryOp(scalar, "==", false);
	}
	
	public MLMatrix $bang$eq(Double scalar) throws IOException, DMLException, ParseException {
		return scalarBinaryOp(scalar, "!=", false);
	}
	
}
