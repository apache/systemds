package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;

public class ConvertRowToCSVString implements Function<Row, String> {
	private static final long serialVersionUID = -2399532576909402664L;

	@Override
	public String call(Row arg0) throws Exception {
		StringBuffer buf = new StringBuffer();
		for(int i = 0; i < arg0.length(); i++) {
			if(i > 0) {
				buf.append(",");
			}
			Double val = new Double(-1);
			try {
				val = Double.parseDouble(arg0.get(i).toString());
			}
			catch(Exception e) {
				throw new Exception("Only double types are supported as input to SystemML");
			}
			buf.append(val.toString());
		}
		return buf.toString();
	}

}
