package com.ibm.bi.dml.api.datasource;

import java.io.Serializable;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.sources.TableScan;
import org.apache.spark.sql.sources.BaseRelation;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;

import com.ibm.bi.dml.api.datasource.functions.GetMLBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

public class MatrixRelation extends BaseRelation implements TableScan, Serializable {
	private static final long serialVersionUID = 1446631085588589012L;
	private SQLContext sqlContext;
	private String file;
	private String format;
	
	public MatrixRelation(SQLContext sqlContext, String file, String format) {
		this.sqlContext = sqlContext;
		this.file = file;
		this.format = format;
	}
	
	@Override
	public RDD<Row> buildScan() {
		// Todo: implement 
		if(format.compareTo("binary") == 0) {
			JavaRDD<Tuple2<MatrixIndexes, MatrixBlock>> rdd = sqlContext.sparkContext().sequenceFile(file, MatrixIndexes.class, MatrixBlock.class).toJavaRDD();
			return rdd.map(new GetMLBlock()).rdd();
		}
		return null;
	}

	@Override
	public StructType schema() {
		return MLBlock.getDefaultSchemaForBinaryBlock();
	}

	@Override
	public SQLContext sqlContext() {
		return sqlContext;
	}
}
