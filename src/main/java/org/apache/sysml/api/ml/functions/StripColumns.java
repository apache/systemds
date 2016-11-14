package org.apache.sysml.api.ml.functions;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.api.java.function.Function;

public class StripColumns implements Function<String, List<String>> {

	private static final long serialVersionUID = -4939143604187818901L;

	@Override
	public List<String> call(String s) throws Exception {
		return Arrays.asList(s.split(" "));
	}

}
