package org.apache.sysml.api.ml.functions;

import org.apache.spark.api.java.function.Function;

public class ConvertRowToDouble implements Function<String, Double> {

	private static final long serialVersionUID = 6003791641019639192L;

	@Override
	public Double call(String str) throws Exception {		
		return Double.parseDouble(str.split(" ")[2]);
	}

}
