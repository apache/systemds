package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.function.Function;

public class ConvertStringToText implements Function<Text, String> {

	private static final long serialVersionUID = 3916028932406746166L;

	@Override
	public String call(Text arg0) throws Exception {
		return arg0.toString();
	}

}
