package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

public class ConvertStringToLongTextPair implements PairFunction<String, LongWritable, Text>{

	private static final long serialVersionUID = 6443041051069809479L;

	@Override
	public Tuple2<LongWritable, Text> call(String arg0) throws Exception {
		return new Tuple2<LongWritable, Text>(new LongWritable(1), new Text(arg0));
	}

}
