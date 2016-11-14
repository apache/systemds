package org.apache.sysml.api.ml.functions;

import java.util.List;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;

public class ConvertListToRow implements Function<List<Object>, Row> {

	private static final long serialVersionUID = 2945331620699893016L;

	@Override
	public Row call(List<Object> arr) throws Exception {
		return RowFactory.create((Double) arr.get(0), Vectors.parse(arr.get(1).toString()), Vectors.parse(arr.get(2).toString()));
	}

}
