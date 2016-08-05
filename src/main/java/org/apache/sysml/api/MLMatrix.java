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
package org.apache.sysml.api;

import java.io.IOException;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.execution.QueryExecution;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.spark.functions.GetMIMBFromRow;
import org.apache.sysml.runtime.instructions.spark.functions.GetMLBlock;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;

/**
 * Experimental API: Might be discontinued in future release
 * 
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

 import org.apache.sysml.api.{MLContext, MLMatrix}
 val ml = new MLContext(sc)
 val mat1 = ml.read(sqlContext, "V_small.csv", "csv")
 val mat2 = ml.read(sqlContext, "W_small.mtx", "binary")
 val result = mat1.transpose() %*% mat2
 result.write("Result_small.mtx", "text")
 
 */
public class MLMatrix extends Dataset<Row> {
	private static final long serialVersionUID = -7005940673916671165L;
	protected static final Log LOG = LogFactory.getLog(DMLScript.class.getName());
	
	protected MatrixCharacteristics mc = null;
	protected MLContext ml = null;
	
	protected MLMatrix(SQLContext sqlContext, LogicalPlan logicalPlan, MLContext ml, Encoder<Row> encoder) {
		super(sqlContext, logicalPlan, encoder);
		this.ml = ml;
	}

	protected MLMatrix(SQLContext sqlContext, QueryExecution queryExecution, MLContext ml, Encoder<Row> encoder) {
		super(sqlContext.sparkSession(), queryExecution, encoder);
		this.ml = ml;
	}
	
	// Only used internally to set a new MLMatrix after one of matrix operations.
	// Not to be used externally.
	protected MLMatrix(Dataset<Row> df, MatrixCharacteristics mc, MLContext ml) throws DMLRuntimeException {
		super(df.sqlContext(), df.logicalPlan(), null); //TODO: Encoder
		this.mc = mc;
		this.ml = ml;
	}
	
	//TODO replace default blocksize
	static String writeStmt = "write(output, \"tmp\", format=\"binary\", rows_in_block=" + OptimizerUtils.DEFAULT_BLOCKSIZE + ", cols_in_block=" + OptimizerUtils.DEFAULT_BLOCKSIZE + ");";
	
	// ------------------------------------------------------------------------------------------------
	
//	/**
//	 * Experimental unstable API: Converts our blocked matrix format to MLLib's format
//	 * @return
//	 */
//	public BlockMatrix toBlockedMatrix() {
//		JavaPairRDD<MatrixIndexes, MatrixBlock> blocks = getRDDLazily(this);
//		RDD<Tuple2<Tuple2<Object, Object>, Matrix>> mllibBlocks = blocks.mapToPair(new GetMLLibBlocks(mc.getRows(), mc.getCols(), mc.getRowsPerBlock(), mc.getColsPerBlock())).rdd();
//		return new BlockMatrix(mllibBlocks, mc.getRowsPerBlock(), mc.getColsPerBlock(), mc.getRows(), mc.getCols());
//	}
	
	// ------------------------------------------------------------------------------------------------
	static MLMatrix createMLMatrix(MLContext ml, SQLContext sqlContext, JavaPairRDD<MatrixIndexes, MatrixBlock> blocks, MatrixCharacteristics mc) throws DMLRuntimeException {
		RDD<Row> rows = blocks.map(new GetMLBlock()).rdd();
		StructType schema = MLBlock.getDefaultSchemaForBinaryBlock();
		return new MLMatrix(sqlContext.createDataFrame(rows.toJavaRDD(), schema), mc, ml);
	}
	
	/**
	 * Convenient method to write a MLMatrix.
	 */
	public void write(String filePath, String format) throws IOException, DMLException {
		ml.reset();
		ml.registerInput("left", this);
		ml.executeScript("left = read(\"\"); output=left; write(output, \"" + filePath + "\", format=\"" + format + "\");");
	}
	
	private double getScalarBuiltinFunctionResult(String fn) throws IOException, DMLException {
		if(fn.equals("nrow") || fn.equals("ncol")) {
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
	public long numRows() throws IOException, DMLException {
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
	public long numCols() throws IOException, DMLException {
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
	
	private MLMatrix matrixBinaryOp(MLMatrix that, String op) throws IOException, DMLException {
		
		if(mc.getRowsPerBlock() != that.mc.getRowsPerBlock() || mc.getColsPerBlock() != that.mc.getColsPerBlock()) {
			throw new DMLRuntimeException("Incompatible block sizes: brlen:" + mc.getRowsPerBlock() + "!=" +  that.mc.getRowsPerBlock() + " || bclen:" + mc.getColsPerBlock() + "!=" + that.mc.getColsPerBlock());
		}
		
		if(op.equals("%*%")) {
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
		return new MLMatrix(this.sqlContext().createDataFrame(rows.toJavaRDD(), schema), mcOut, ml);
	}
	
	private MLMatrix scalarBinaryOp(Double scalar, String op, boolean isScalarLeft) throws IOException, DMLException {
		ml.reset();
		ml.registerInput("left", this);
		ml.registerOutput("output");
		MLOutput out = ml.executeScript(getScalarBinaryScript(op, scalar, isScalarLeft));
		RDD<Row> rows = out.getBinaryBlockedRDD("output").map(new GetMLBlock()).rdd();
		StructType schema = MLBlock.getDefaultSchemaForBinaryBlock();
		MatrixCharacteristics mcOut = out.getMatrixCharacteristics("output");
		return new MLMatrix(this.sqlContext().createDataFrame(rows.toJavaRDD(), schema), mcOut, ml);
	}
	
	// ---------------------------------------------------
	// Simple operator loading but doesnot utilize the optimizer
	
	public MLMatrix $greater(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, ">");
	}
	
	public MLMatrix $less(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "<");
	}
	
	public MLMatrix $greater$eq(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, ">=");
	}
	
	public MLMatrix $less$eq(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "<=");
	}
	
	public MLMatrix $eq$eq(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "==");
	}
	
	public MLMatrix $bang$eq(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "!=");
	}
	
	public MLMatrix $up(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "^");
	}
	
	public MLMatrix exp(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "^");
	}
	
	public MLMatrix $plus(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "+");
	}
	
	public MLMatrix add(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "+");
	}
	
	public MLMatrix $minus(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "-");
	}
	
	public MLMatrix minus(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "-");
	}
	
	public MLMatrix $times(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "*");
	}
	
	public MLMatrix elementWiseMultiply(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "*");
	}
	
	public MLMatrix $div(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "/");
	}
	
	public MLMatrix divide(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "/");
	}
	
	public MLMatrix $percent$div$percent(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "%/%");
	}
	
	public MLMatrix integerDivision(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "%/%");
	}
	
	public MLMatrix $percent$percent(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "%%");
	}
	
	public MLMatrix modulus(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "%%");
	}
	
	public MLMatrix $percent$times$percent(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "%*%");
	}
	
	public MLMatrix multiply(MLMatrix that) throws IOException, DMLException {
		return matrixBinaryOp(that, "%*%");
	}
	
	public MLMatrix transpose() throws IOException, DMLException {
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
		return new MLMatrix(this.sqlContext().createDataFrame(rows.toJavaRDD(), schema), mcOut, ml);
	}
	
	// TODO: For 'scalar op matrix' operations: Do implicit conversions 
	public MLMatrix $plus(Double scalar) throws IOException, DMLException {
		return scalarBinaryOp(scalar, "+", false);
	}
	
	public MLMatrix add(Double scalar) throws IOException, DMLException {
		return scalarBinaryOp(scalar, "+", false);
	}
	
	public MLMatrix $minus(Double scalar) throws IOException, DMLException {
		return scalarBinaryOp(scalar, "-", false);
	}
	
	public MLMatrix minus(Double scalar) throws IOException, DMLException {
		return scalarBinaryOp(scalar, "-", false);
	}
	
	public MLMatrix $times(Double scalar) throws IOException, DMLException {
		return scalarBinaryOp(scalar, "*", false);
	}
	
	public MLMatrix elementWiseMultiply(Double scalar) throws IOException, DMLException {
		return scalarBinaryOp(scalar, "*", false);
	}
	
	public MLMatrix $div(Double scalar) throws IOException, DMLException {
		return scalarBinaryOp(scalar, "/", false);
	}
	
	public MLMatrix divide(Double scalar) throws IOException, DMLException {
		return scalarBinaryOp(scalar, "/", false);
	}
	
	public MLMatrix $greater(Double scalar) throws IOException, DMLException {
		return scalarBinaryOp(scalar, ">", false);
	}
	
	public MLMatrix $less(Double scalar) throws IOException, DMLException {
		return scalarBinaryOp(scalar, "<", false);
	}
	
	public MLMatrix $greater$eq(Double scalar) throws IOException, DMLException {
		return scalarBinaryOp(scalar, ">=", false);
	}
	
	public MLMatrix $less$eq(Double scalar) throws IOException, DMLException {
		return scalarBinaryOp(scalar, "<=", false);
	}
	
	public MLMatrix $eq$eq(Double scalar) throws IOException, DMLException {
		return scalarBinaryOp(scalar, "==", false);
	}
	
	public MLMatrix $bang$eq(Double scalar) throws IOException, DMLException {
		return scalarBinaryOp(scalar, "!=", false);
	}
	
}
