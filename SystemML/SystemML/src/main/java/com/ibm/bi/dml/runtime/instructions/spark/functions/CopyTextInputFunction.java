package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

public class CopyTextInputFunction implements PairFunction<Tuple2<LongWritable, Text>,LongWritable, Text> 
{
	private static final long serialVersionUID = -196553327495233360L;

	public CopyTextInputFunction(  ) {
	
	}

	@Override
	public Tuple2<LongWritable, Text> call(
		Tuple2<LongWritable, Text> arg0) throws Exception {
		LongWritable lw = new LongWritable(arg0._1().get());
		Text txt = new Text(arg0._2());
		return new Tuple2<LongWritable,Text>(lw, txt);
	}
}