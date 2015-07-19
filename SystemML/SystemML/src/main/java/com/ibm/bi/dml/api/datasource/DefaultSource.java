package com.ibm.bi.dml.api.datasource;

import java.io.IOException;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.sources.BaseRelation;
import org.apache.spark.sql.sources.CreatableRelationProvider;
import org.apache.spark.sql.sources.RelationProvider;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.ibm.bi.dml.api.datasource.functions.GetMIMBFromRow;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.spark.data.RDDObject;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

import scala.collection.immutable.Map;

public class DefaultSource implements RelationProvider, CreatableRelationProvider {

	@Override
	public BaseRelation createRelation(SQLContext sqlContext, Map<String, String> parameters) {
		if(parameters.contains("file") && parameters.contains("format")) {
			return new MatrixRelation(sqlContext, parameters.get("file").get(), parameters.get("format").get());
		}
		
		throw new RuntimeException("Incorrect parameters passed to createRelation");
	}

	@Override
	public BaseRelation createRelation(SQLContext sqlContext, SaveMode mode,
			Map<String, String> parameters, DataFrame df) {
		if(parameters.contains("file") && parameters.contains("format")) {
			boolean save = false;
			String filePath = parameters.get("file").get();
			String fileFormat = parameters.get("format").get();
			Path path = new Path(filePath);
			FileSystem fs;
			try {
				fs = path.getFileSystem(sqlContext.sparkContext().hadoopConfiguration());
			} catch (IOException e) {
				throw new RuntimeException("Error accessing HDFS filesystem:"  + e.getMessage());
			}
			switch(mode) {
				case Append: 
					throw new RuntimeException("Appending to file is not supported");
					
				case ErrorIfExists:
					try {
						if(fs.exists(path)) {
							throw new RuntimeException("The path already exists:" + filePath);
						}
						save = true;
					} catch (IOException e1) {
						throw new RuntimeException("Error accessing HDFS filesystem:"  + e1.getMessage());
					}
					break;
					
				case Overwrite:
					try {
						if(fs.exists(path)) {
							fs.delete(path, true);
						}
						save = true;
					} catch (IOException e1) {
						throw new RuntimeException("Error accessing HDFS filesystem:"  + e1.getMessage());
					}
					break;
					
				case Ignore:
					break;
					
				default:
					throw new RuntimeException("Unsupported save mode:" + mode.name());
			}
			
			if(save) {
				JavaPairRDD<MatrixIndexes, MatrixBlock> binaryRDD = df.rdd().toJavaRDD().mapToPair(new GetMIMBFromRow());
				RDDObject rddObject = new RDDObject(binaryRDD, "output");
				if(fileFormat.compareTo("binary") == 0) {
					try {
						MapReduceTool.writeMetaDataFile(filePath + ".mtd", ValueType.DOUBLE, new MatrixCharacteristics(10, 5, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize), OutputInfo.BinaryBlockOutputInfo);
					} catch (IOException e) {
						throw new RuntimeException(e);
					}
					SparkExecutionContext.writeRDDtoHDFS(rddObject, filePath, OutputInfo.BinaryBlockOutputInfo);
					// binaryRDD.saveAsHadoopFile(filePath, MatrixIndexes.class, MatrixBlock.class, SequenceFileOutputFormat.class);
				}
				else if(fileFormat.compareTo("csv") == 0) {
					SparkExecutionContext.writeRDDtoHDFS(rddObject, filePath, OutputInfo.CSVOutputInfo);
				}
				else if(fileFormat.compareTo("text") == 0) {
					SparkExecutionContext.writeRDDtoHDFS(rddObject, filePath, OutputInfo.TextCellOutputInfo);
				}
				else {
					throw new RuntimeException("Unsupported file format:" + fileFormat);
				}
			}
			
			return new MatrixRelation(sqlContext, filePath, fileFormat);
		}
		throw new RuntimeException("Incorrect parameters passed to createRelation");
	}
	
	

}
