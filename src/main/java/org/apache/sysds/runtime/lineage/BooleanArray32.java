package org.apache.sysds.runtime.lineage;

public class BooleanArray32 {
	private int _value;
	
	public BooleanArray32(int value){
		_value = value;
	}
	
	public boolean get(int pos) {
		return (_value & (1 << pos)) != 0;
	}
	
	public void set(int pos, boolean value) {
		int mask = 1 << pos;
		_value = (_value & ~mask) | (value ? mask : 0);
	}
	
	public int getValue() { return _value; }
	
	public void setValue(int value) { _value = value; }
} 
