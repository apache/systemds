package dml.runtime.functionobjects;

public class GreaterThanEqualsReturnDouble extends ValueFunction {

	public static GreaterThanEqualsReturnDouble singleObj = null;

	private GreaterThanEqualsReturnDouble() {
		// nothing to do here
	}
	
	public static GreaterThanEqualsReturnDouble getGreaterThanEqualsReturnDoubleFnObject() {
		if ( singleObj == null )
			singleObj = new GreaterThanEqualsReturnDouble();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public double execute(double in1, double in2) {
		return (Double.compare(in1, in2) >= 0 ? 1.0 : 0.0);
	}

}
