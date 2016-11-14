package org.apache.sysml.api.ml.functions;

import java.util.List;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;

import scala.collection.JavaConversions;

public class ConvertRowToListOfObjects implements Function<Row, List<Object>> {

	private static final long serialVersionUID = -9008851379299137955L;

	@Override
	public List<Object> call(Row row) throws Exception {
		return JavaConversions.seqAsJavaList(row.toSeq());
	}

}
