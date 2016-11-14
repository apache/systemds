package org.apache.sysml.api.ml.functions;

import org.apache.sysml.api.ml.feature.DummyCodeGenerator;
import java.util.ArrayList;
import java.util.List;
import org.apache.spark.api.java.function.Function;


public class AddMinMaxLabelValue implements Function<List<Object>, List<Object>> {
	
	private static final long serialVersionUID = -4811728115034184293L;
	private final double minVal = DummyCodeGenerator.getMinLabelVal();
	private final double maxVal = DummyCodeGenerator.getMaxLabelVal();

	@Override
	public List<Object> call(List<Object> s) throws Exception {
		List<Object> result = new ArrayList<Object>();
		
		result.addAll(s);
		result.add((Object) minVal);
		result.add((Object) maxVal);
		
		return result;
	}
}
