package dml.runtime.functionobjects;

public class GreaterThan extends ValueFunction {
	
	public static GreaterThan singleObj = null;

	private GreaterThan() {
		// nothing to do here
	}
	
	public static GreaterThan getGreaterThanFnObject() {
		if ( singleObj == null )
			singleObj = new GreaterThan();
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

	@Override
	public boolean compare(double in1, double in2) {
		return (Double.compare(in1, in2) > 0);
	}

	@Override
	public boolean compare(int in1, int in2) {
		return (in1 > in2);
	}

	@Override
	public boolean compare(double in1, int in2) {
		return (Double.compare(in1, in2) > 0);
	}

	@Override
	public boolean compare(int in1, double in2) {
		return (Double.compare(in1, in2) > 0);
	}

}
