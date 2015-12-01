package org.apache.sysml.api.ml.functions;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;

public class ConvertVectorToDouble implements Function<Row, Double> {

	private static final long serialVersionUID = -6612447783777073929L;

	@Override
	public Double call(Row row) throws Exception {
		
		return row.getDouble(0);
	}

}
