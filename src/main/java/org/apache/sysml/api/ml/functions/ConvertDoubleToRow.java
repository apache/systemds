package org.apache.sysml.api.ml.functions;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;

public class ConvertDoubleToRow implements Function<Double, Row> {

	private static final long serialVersionUID = 6873705400902186639L;

	@Override
	public Row call(Double value) throws Exception {
		return RowFactory.create(value);
	}

}
