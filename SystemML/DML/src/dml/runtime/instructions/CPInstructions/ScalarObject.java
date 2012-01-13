package dml.runtime.instructions.CPInstructions;

public abstract class ScalarObject extends Data {

	private String _name;

	public ScalarObject(String name) {
		_name = name;
	}

	public String getName() {
		return _name;
	}

	public abstract Object getValue();

	public abstract int getIntValue();

	public abstract double getDoubleValue();

	public abstract boolean getBooleanValue();

	public abstract String getStringValue();
	
	public abstract long getLongValue();
}
