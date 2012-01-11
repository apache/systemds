package dml.runtime.functionobjects;

public class LessThanReturnDouble extends ValueFunction {
	
	public static LessThanReturnDouble singleObj = null;

	private LessThanReturnDouble() {
		// nothing to do here
	}
	
	public static LessThanReturnDouble getLessThanReturnDoubleFnObject() {
		if ( singleObj == null )
			singleObj = new LessThanReturnDouble();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public double execute(double in1, double in2) {
		return (Double.compare(in1, in2) < 0 ? 1.0 : 0.0);
	}
}
