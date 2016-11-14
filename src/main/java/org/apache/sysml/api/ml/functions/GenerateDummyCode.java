package org.apache.sysml.api.ml.functions;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.function.Function;

public class GenerateDummyCode implements Function<List<Object>, List<Object>> {
	
	private static final long serialVersionUID = 8288904231567560245L;

	@Override
	public List<Object> call(List<Object> arr) throws Exception {
		double value = (Double) arr.get(0);
		double minLabelValue = (Double) arr.get(2);
		double maxLabelValue = (Double) arr.get(3);
		List<Object> result = new ArrayList<Object>();
		List<Object> dummy = new ArrayList<Object>();
		result.add(arr.get(0));
		result.add(arr.get(1));
		
		for (int i = (int) minLabelValue; i <= (int) maxLabelValue; i++)
		{
			if (i == value)
				dummy.add(1.0);
			else
				dummy.add(0);
		}
		
		result.add(dummy);
		
		return result;
	}
}
