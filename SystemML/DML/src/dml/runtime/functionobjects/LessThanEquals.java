package dml.runtime.functionobjects;

public class LessThanEquals extends ValueFunction {

	public static LessThanEquals singleObj = null;

	private LessThanEquals() {
		// nothing to do here
	}
	
	public static LessThanEquals getLessThanEqualsFnObject() {
		if ( singleObj == null )
			singleObj = new LessThanEquals();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public boolean compare(double in1, double in2) {
		return (Double.compare(in1, in2) <= 0);
	}

	@Override
	public boolean compare(int in1, int in2) {
		return (in1 <= in2);
	}

	@Override
	public boolean compare(double in1, int in2) {
		return (Double.compare(in1, in2) <= 0);
	}

	@Override
	public boolean compare(int in1, double in2) {
		return (Double.compare(in1, in2) <= 0);
	}
}
