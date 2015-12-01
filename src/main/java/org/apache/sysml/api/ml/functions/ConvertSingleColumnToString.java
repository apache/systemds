package org.apache.sysml.api.ml.functions;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;

public class ConvertSingleColumnToString implements Function<Row, String> {

	private static final long serialVersionUID = -499763403738768970L;

	@Override
	public String call(Row row) throws Exception {
		return row.apply(0).toString();
	}
}
	
