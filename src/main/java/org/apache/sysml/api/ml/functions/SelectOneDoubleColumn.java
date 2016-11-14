package org.apache.sysml.api.ml.functions;

import java.util.List;

import org.apache.spark.api.java.function.Function;

public class SelectOneDoubleColumn implements Function<List<String>, Double> {

	private static final long serialVersionUID = 3611521546505801707L;

	@Override
	public Double call(List<String> arr) throws Exception {
		return Double.parseDouble(arr.get(2));
	}

}
