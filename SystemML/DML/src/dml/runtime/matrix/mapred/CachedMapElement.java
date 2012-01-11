package dml.runtime.matrix.mapred;

public abstract class CachedMapElement{

	public abstract void set(CachedMapElement elem);
	public abstract CachedMapElement duplicate();
}
