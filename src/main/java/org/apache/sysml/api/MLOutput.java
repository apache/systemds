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

import java.util.Map;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.StructType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.spark.functions.GetMLBlock;
import org.apache.sysml.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;

/**
 * This is a simple container object that returns the output of execute from MLContext 
 *
 */
public class MLOutput {
	
	Map<String, JavaPairRDD<?,?>> _outputs;
	private Map<String, MatrixCharacteristics> _outMetadata = null;
	
	public MLOutput(Map<String, JavaPairRDD<?,?>> outputs, Map<String, MatrixCharacteristics> outMetadata) {
		this._outputs = outputs;
		this._outMetadata = outMetadata;
	}
	
	public MatrixBlock getMatrixBlock(String varName) throws DMLRuntimeException {
		MatrixCharacteristics mc = getMatrixCharacteristics(varName);
		// The matrix block is always pushed to an RDD and then we do collect
		// We can later avoid this by returning symbol table rather than "Map<String, JavaPairRDD<MatrixIndexes,MatrixBlock>> _outputs"
		return SparkExecutionContext.toMatrixBlock(getBinaryBlockedRDD(varName), (int) mc.getRows(), (int) mc.getCols(), 
				mc.getRowsPerBlock(), mc.getColsPerBlock(), mc.getNonZeros());
	}
	
	@SuppressWarnings("unchecked")
	public JavaPairRDD<MatrixIndexes,MatrixBlock> getBinaryBlockedRDD(String varName) throws DMLRuntimeException {
		if(_outputs.containsKey(varName)) {
			return (JavaPairRDD<MatrixIndexes,MatrixBlock>) _outputs.get(varName);
		}
		throw new DMLRuntimeException("Variable " + varName + " not found in the outputs.");
	}
	
	@SuppressWarnings("unchecked")
	public JavaPairRDD<Long,FrameBlock> getFrameBinaryBlockedRDD(String varName) throws DMLRuntimeException {
		if(_outputs.containsKey(varName)) {
			return (JavaPairRDD<Long,FrameBlock>)_outputs.get(varName);
		}
		throw new DMLRuntimeException("Variable " + varName + " not found in the outputs.");
	}
	
	public MatrixCharacteristics getMatrixCharacteristics(String varName) throws DMLRuntimeException {
		if(_outputs.containsKey(varName)) {
			return _outMetadata.get(varName);
		}
		throw new DMLRuntimeException("Variable " + varName + " not found in the output symbol table.");
	}
	
	/**
	 * Note, the output DataFrame has an additional column ID.
	 * An easy way to get DataFrame without ID is by df.drop("__INDEX")
	 * @param sqlContext
	 * @param varName
	 * @return
	 * @throws DMLRuntimeException
	 */
	public DataFrame getDF(SQLContext sqlContext, String varName) throws DMLRuntimeException {
		if(sqlContext == null) {
			throw new DMLRuntimeException("SQLContext is not created.");
		}
		JavaPairRDD<MatrixIndexes,MatrixBlock> rdd = getBinaryBlockedRDD(varName);
		if(rdd != null) {
			MatrixCharacteristics mc = _outMetadata.get(varName);
			return RDDConverterUtils.binaryBlockToDataFrame(sqlContext, rdd, mc, false);
		}
		throw new DMLRuntimeException("Variable " + varName + " not found in the output symbol table.");
	}
	
	/**
	 * 
	 * @param sqlContext
	 * @param varName
	 * @param outputVector if true, returns DataFrame with two column: ID and org.apache.spark.mllib.linalg.Vector
	 * @return
	 * @throws DMLRuntimeException
	 */
	public DataFrame getDF(SQLContext sqlContext, String varName, boolean outputVector) throws DMLRuntimeException {
		if(sqlContext == null) {
			throw new DMLRuntimeException("SQLContext is not created.");
		}
		if(outputVector) {
			JavaPairRDD<MatrixIndexes,MatrixBlock> rdd = getBinaryBlockedRDD(varName);
			if(rdd != null) {
				MatrixCharacteristics mc = _outMetadata.get(varName);
				return RDDConverterUtils.binaryBlockToDataFrame(sqlContext, rdd, mc, true);
			}
			throw new DMLRuntimeException("Variable " + varName + " not found in the output symbol table.");
		}
		else {
			return getDF(sqlContext, varName);
		}
		
	}
	
	/**
	 * This methods improves the performance of MLPipeline wrappers.
	 * @param sqlContext
	 * @param varName
	 * @param range range is inclusive
	 * @return
	 * @throws DMLRuntimeException
	 */
	public DataFrame getDF(SQLContext sqlContext, String varName, MatrixCharacteristics mc) 
		throws DMLRuntimeException 
	{
		if(sqlContext == null)
			throw new DMLRuntimeException("SQLContext is not created.");
			
		JavaPairRDD<MatrixIndexes,MatrixBlock> binaryBlockRDD = getBinaryBlockedRDD(varName);
		return RDDConverterUtils.binaryBlockToDataFrame(sqlContext, binaryBlockRDD, mc, true);
	}
	
	public JavaRDD<String> getStringRDD(String varName, String format) throws DMLRuntimeException {
		if(format.equals("text")) {
			JavaPairRDD<MatrixIndexes, MatrixBlock> binaryRDD = getBinaryBlockedRDD(varName);
			MatrixCharacteristics mcIn = getMatrixCharacteristics(varName); 
			return RDDConverterUtils.binaryBlockToTextCell(binaryRDD, mcIn);
		}
		else {
			throw new DMLRuntimeException("The output format:" + format + " is not implemented yet.");
		}
	}
	
	public JavaRDD<String> getStringFrameRDD(String varName, String format, CSVFileFormatProperties fprop ) throws DMLRuntimeException {
		//TODO MB: note that on construction of MLOutput only matrix binary blocks are passed, and 
		//hence we will never find a frame binary block in the outputs.
		JavaPairRDD<Long, FrameBlock> binaryRDD = getFrameBinaryBlockedRDD(varName);
		MatrixCharacteristics mcIn = getMatrixCharacteristics(varName); 
		if(format.equals("csv")) {
			return FrameRDDConverterUtils.binaryBlockToCsv(binaryRDD, mcIn, fprop, false);
		}
		else if(format.equals("text")) {
			return FrameRDDConverterUtils.binaryBlockToTextCell(binaryRDD, mcIn);
		}
		else {
			throw new DMLRuntimeException("The output format:" + format + " is not implemented yet.");
		}
		
	}
	
	public DataFrame getDataFrameRDD(String varName, JavaSparkContext jsc) throws DMLRuntimeException {
		//TODO MB: note that on construction of MLOutput only matrix binary blocks are passed, and 
		//hence we will never find a frame binary block in the outputs.
		JavaPairRDD<Long, FrameBlock> binaryRDD = getFrameBinaryBlockedRDD(varName);
		MatrixCharacteristics mcIn = getMatrixCharacteristics(varName);
		return FrameRDDConverterUtils.binaryBlockToDataFrame(new SQLContext(jsc), binaryRDD, mcIn, null);
	}
	
	public MLMatrix getMLMatrix(MLContext ml, SQLContext sqlContext, String varName) throws DMLRuntimeException {
		if(sqlContext == null) {
			throw new DMLRuntimeException("SQLContext is not created.");
		}
		else if(ml == null) {
			throw new DMLRuntimeException("MLContext is not created.");
		}
		JavaPairRDD<MatrixIndexes,MatrixBlock> rdd = getBinaryBlockedRDD(varName);
		if(rdd != null) {
			MatrixCharacteristics mc = getMatrixCharacteristics(varName);
			StructType schema = MLBlock.getDefaultSchemaForBinaryBlock();
			return new MLMatrix(sqlContext.createDataFrame(rdd.map(new GetMLBlock()).rdd(), schema), mc, ml);
		}
		throw new DMLRuntimeException("Variable " + varName + " not found in the output symbol table.");
	}
}