package com.ibm.bi.dml.api;

import java.io.IOException;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SQLContext.QueryExecution;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ConvertMLLibBlocksToBinaryBlocks;
import com.ibm.bi.dml.runtime.instructions.spark.functions.GetMIMBFromRow;
import com.ibm.bi.dml.runtime.instructions.spark.functions.GetMLBlock;
import com.ibm.bi.dml.runtime.instructions.spark.functions.GetMLLibBlocks;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.distributed.BlockMatrix;

/**
 * This class serves four purposes:
 * 1. It allows SystemML to fit nicely in MLPipeline by reducing number of reblocks.
 * 2. It allows users to easily read and write matrices without worrying 
 * too much about format, metadata and type of underlying RDDs.
 * 3. It provides mechanism to convert to and from MLLib's BlockedMatrix format
 * 4. It provides off-the-shelf library for Distributed Blocked Matrix and reduces learning curve for using SystemML.
 * However, it is important to know that it is easy to abuse this off-the-shelf library and think it as replacement
 * to writing DML, which it is not. It does not provide any optimization between calls. A simple example
 * of the optimization that is conveniently skipped is: (t(m) %*% m)).
 * Also, note that this library is not thread-safe. The operator precedence is not exactly same as DML (as the precedence is
 * enforced by scala compiler), so please use appropriate brackets to enforce precedence. 

 import com.ibm.bi.dml.api.{MLContext, MLMatrix}
 val ml = new MLContext(sc)
 val mat1 = MLMatrix.create(sqlContext, "V_small.csv", "csv")
 val mat2 = MLMatrix.create(sqlContext, "W_small.mtx", "binary")
 val result = mat1.transpose() %*% mat2
 result.write("Result_small.mtx", "text")
 
 */
public class MLMatrix extends DataFrame {
	private static final long serialVersionUID = -7005940673916671165L;
	protected static final Log LOG = LogFactory.getLog(DMLScript.class.getName());
	
	protected MatrixCharacteristics mc = null;
	
	protected MLMatrix(SQLContext sqlContext, LogicalPlan logicalPlan) {
		super(sqlContext, logicalPlan);
	}

	protected MLMatrix(SQLContext sqlContext, QueryExecution queryExecution) {
		super(sqlContext, queryExecution);
	}
	
	// Only used internally to set a new MLMatrix after one of matrix operations.
	// Not to be used externally.
	protected MLMatrix(DataFrame df, MatrixCharacteristics mc) throws DMLRuntimeException {
		super(df.sqlContext(), df.logicalPlan());
		this.mc = mc;
	}
	
	private static String writeStmt = "write(output, \"tmp\", format=\"binary\", rows_in_block=" + DMLTranslator.DMLBlockSize + ", cols_in_block=" + DMLTranslator.DMLBlockSize + ");";
	
	// TODO: Add additional create to provide sep, missing values, etc. for CSV
	public static MLMatrix create(SQLContext sqlContext, String filePath, String format) throws IOException, DMLException, ParseException {
		MLContext ml = checkAndGetMLContext();
		ml.reset();
		ml.registerOutput("output");
		MLOutput out = ml.executeScript("output = read(\"" + filePath + "\", format=\"" + format + "\"); " + writeStmt);
		JavaPairRDD<MatrixIndexes, MatrixBlock> blocks = out.getBinaryBlockedRDD("output");
		MatrixCharacteristics mcOut = out.getMatrixCharacteristics("output");
		return createMLMatrix(sqlContext, blocks, mcOut);
	}
	
	// ------------------------------------------------------------------------------------------------
	// TODO: Test this in different scenarios: sparse/dense/mixed
	public static MLMatrix create(SQLContext sqlContext, BlockMatrix mllibMatrix) throws DMLRuntimeException {
		long nnz = -1; // TODO: Find number of non-zeros from mllibMatrix ... This is important !!
		
		JavaPairRDD<Tuple2<Object, Object>, Matrix> mllibBlocks = JavaPairRDD.fromJavaRDD(mllibMatrix.blocks().toJavaRDD());
		long rlen = mllibMatrix.numRows(); long clen = mllibMatrix.numCols();
		int brlen = mllibMatrix.numRowBlocks();
		int bclen = mllibMatrix.numColBlocks();
		if(mllibMatrix.numRowBlocks() != DMLTranslator.DMLBlockSize && mllibMatrix.numColBlocks() != DMLTranslator.DMLBlockSize) {
			// TODO: Show warning as this will require an reblock later 
			// OR perform reblock while creating itself
		}
		
		JavaPairRDD<MatrixIndexes, MatrixBlock> blocks = mllibBlocks
				.mapToPair(new ConvertMLLibBlocksToBinaryBlocks(rlen, clen, brlen, bclen));
		
		MatrixCharacteristics mc = new MatrixCharacteristics(rlen, clen, brlen, bclen, nnz);
		return createMLMatrix(sqlContext, blocks, mc);
	}
	
	/**
	 * Converts our blocked matrix format to MLLib's format (Not tested!!)
	 * @return
	 */
	public BlockMatrix toBlockedMatrix() {
		JavaPairRDD<MatrixIndexes, MatrixBlock> blocks = getRDDLazily(this);
		RDD<Tuple2<Tuple2<Object, Object>, Matrix>> mllibBlocks = blocks.mapToPair(new GetMLLibBlocks(mc.getRows(), mc.getCols(), mc.getRowsPerBlock(), mc.getColsPerBlock())).rdd();
		return new BlockMatrix(mllibBlocks, mc.getRowsPerBlock(), mc.getColsPerBlock(), mc.getRows(), mc.getCols());
	}
	
	// ------------------------------------------------------------------------------------------------
		
	private static MLMatrix createMLMatrix(SQLContext sqlContext, JavaPairRDD<MatrixIndexes, MatrixBlock> blocks, MatrixCharacteristics mc) throws DMLRuntimeException {
		RDD<Row> rows = blocks.map(new GetMLBlock()).rdd();
		StructType schema = MLBlock.getDefaultSchemaForBinaryBlock();
		return new MLMatrix(sqlContext.createDataFrame(rows.toJavaRDD(), schema), mc);
	}
	
	/**
	 * Convenient method to write a MLMatrix.
	 */
	public void write(String filePath, String format) throws IOException, DMLException, ParseException {
		MLContext ml = checkAndGetMLContext();
		ml.reset();
		ml.registerInput("left", this);
		ml.executeScript("left = read(\"\"); output=left; write(output, \"" + filePath + "\", format=\"" + format + "\");");
	}
	
	private static MLContext checkAndGetMLContext() throws DMLRuntimeException {
		if(MLContext.getCurrentMLContext() == null) {
			throw new DMLRuntimeException("ERROR: No MLContext is created for this session");
		}
		return MLContext.getCurrentMLContext();
	}
	
	private double getScalarBuiltinFunctionResult(String fn) throws IOException, DMLException, ParseException {
		if(fn.compareTo("nrow") == 0 || fn.compareTo("ncol") == 0) {
			MLContext ml = checkAndGetMLContext();
			ml.reset();
			ml.registerInput("left", getRDDLazily(this), mc.getRows(), mc.getCols(), mc.getRowsPerBlock(), mc.getColsPerBlock(), mc.getNonZeros());
			ml.registerOutput("output");
			String script = "left = read(\"\");"
					+ "val = " + fn + "(left); "
					+ "output = matrix(val, rows=1, cols=1); "
					+ writeStmt;
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
		if(mc.rowsKnown()) {
			return mc.getRows();
		}
		else {
			return  (long) getScalarBuiltinFunctionResult("nrow");
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
		if(mc.colsKnown()) {
			return mc.getCols();
		}
		else {
			return (long) getScalarBuiltinFunctionResult("ncol");
		}
	}
	
	public int rowsPerBlock() {
		return mc.getRowsPerBlock();
	}
	
	public int colsPerBlock() {
		return mc.getColsPerBlock();
	}
	
	private String getScript(String binaryOperator) {
		return 	"left = read(\"\");"
				+ "right = read(\"\");"
				+ "output = left " + binaryOperator + " right; "
				+ writeStmt;
	}
	
	private String getScalarBinaryScript(String binaryOperator, double scalar, boolean isScalarLeft) {
		if(isScalarLeft) {
			return 	"left = read(\"\");"
					+ "output = " + scalar + " " + binaryOperator + " left ;"
					+ writeStmt;
		}
		else {
			return 	"left = read(\"\");"
				+ "output = left " + binaryOperator + " " + scalar + ";"
				+ writeStmt;
		}
	}
	
	static JavaPairRDD<MatrixIndexes, MatrixBlock> getRDDLazily(MLMatrix mat) {
		return mat.rdd().toJavaRDD().mapToPair(new GetMIMBFromRow());
	}
	
	private MLMatrix matrixBinaryOp(MLMatrix that, String op) throws IOException, DMLException, ParseException {
		MLContext ml = checkAndGetMLContext();
		
		if(mc.getRowsPerBlock() != that.mc.getRowsPerBlock() || mc.getColsPerBlock() != that.mc.getColsPerBlock()) {
			throw new DMLRuntimeException("Incompatible block sizes: brlen:" + mc.getRowsPerBlock() + "!=" +  that.mc.getRowsPerBlock() + " || bclen:" + mc.getColsPerBlock() + "!=" + that.mc.getColsPerBlock());
		}
		
		if(op.compareTo("%*%") == 0) {
			if(mc.getCols() != that.mc.getRows()) {
				throw new DMLRuntimeException("Dimensions mismatch:" + mc.getCols() + "!=" +  that.mc.getRows());
			}
		}
		else {
			if(mc.getRows() != that.mc.getRows() || mc.getCols() != that.mc.getCols()) {
				throw new DMLRuntimeException("Dimensions mismatch:" + mc.getRows() + "!=" +  that.mc.getRows() + " || " + mc.getCols() + "!=" + that.mc.getCols());
			}
		}
		
		ml.reset();
		ml.registerInput("left", this);
		ml.registerInput("right", that);
		ml.registerOutput("output");
		MLOutput out = ml.executeScript(getScript(op));
		RDD<Row> rows = out.getBinaryBlockedRDD("output").map(new GetMLBlock()).rdd();
		StructType schema = MLBlock.getDefaultSchemaForBinaryBlock();
		MatrixCharacteristics mcOut = out.getMatrixCharacteristics("output");
		return new MLMatrix(this.sqlContext().createDataFrame(rows.toJavaRDD(), schema), mcOut);
	}
	
	private MLMatrix scalarBinaryOp(Double scalar, String op, boolean isScalarLeft) throws IOException, DMLException, ParseException {
		MLContext ml = checkAndGetMLContext();
		
		ml.reset();
		ml.registerInput("left", this);
		ml.registerOutput("output");
		MLOutput out = ml.executeScript(getScalarBinaryScript(op, scalar, isScalarLeft));
		RDD<Row> rows = out.getBinaryBlockedRDD("output").map(new GetMLBlock()).rdd();
		StructType schema = MLBlock.getDefaultSchemaForBinaryBlock();
		MatrixCharacteristics mcOut = out.getMatrixCharacteristics("output");
		return new MLMatrix(this.sqlContext().createDataFrame(rows.toJavaRDD(), schema), mcOut);
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
		MLContext ml = checkAndGetMLContext();
		
		ml.reset();
		ml.registerInput("left", this);
		ml.registerOutput("output");
		String script = "left = read(\"\");"
				+ "output = t(left); "
				+ writeStmt;
		MLOutput out = ml.executeScript(script);
		RDD<Row> rows = out.getBinaryBlockedRDD("output").map(new GetMLBlock()).rdd();
		StructType schema = MLBlock.getDefaultSchemaForBinaryBlock();
		MatrixCharacteristics mcOut = out.getMatrixCharacteristics("output");
		return new MLMatrix(this.sqlContext().createDataFrame(rows.toJavaRDD(), schema), mcOut);
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
