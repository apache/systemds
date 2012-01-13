package dml.runtime.functionobjects;

public class GreaterThanReturnDouble extends ValueFunction {
	
	public static GreaterThanReturnDouble singleObj = null;

	private GreaterThanReturnDouble() {
		// nothing to do here
	}
	
	public static GreaterThanReturnDouble getGreaterThanReturnDoubleFnObject() {
		if ( singleObj == null )
			singleObj = new GreaterThanReturnDouble();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public double execute(double in1, double in2) {
		return (Double.compare(in1, in2) > 0 ? 1.0 : 0.0);
	}
}
