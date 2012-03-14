package dml.runtime.matrix.mapred;

import java.util.HashMap;
import java.util.Set;
import java.util.Vector;
import java.util.Map.Entry;


public class CachedMap<T extends CachedMapElement> {

	protected HashMap<Byte, Integer> map=new HashMap<Byte, Integer>();
	protected Vector<T> cache=new Vector<T>();
	protected int numValid=0;
	
	public CachedMap()
	{}
	
	@SuppressWarnings("unchecked")
	public T set(Byte index, T value)
	{
		if(numValid<cache.size())	
			cache.elementAt(numValid).set(value);
		else
			cache.add((T) value.duplicate());
		map.put(index, numValid);
		numValid++;
		return cache.get(numValid-1);
	}
	
	public void reset()
	{
		numValid=0;
		map.clear();
	}
	
	public void remove(byte index)
	{
		Integer cacheIndex=map.remove(index);
		if(cacheIndex==null)
			return;
		
		if(cacheIndex==numValid-1)
		{
			numValid--;
			return;
		}
		//swap the last element and the element to remove
		T lastElem=cache.elementAt(numValid-1);
		cache.set(numValid-1, cache.get(cacheIndex));
		cache.set(cacheIndex, lastElem);
		//remap the indexes
		for(Entry<Byte, Integer> entry: map.entrySet())
		{
			if(entry.getValue()==numValid-1)
			{
				entry.setValue(cacheIndex);
				break;
			}
		}
		numValid--;
	}
	
	public T get(byte index)
	{
		Integer cacheIndex=map.get(index);
		if(cacheIndex==null)
			return null;
		else
			return cache.elementAt(cacheIndex);
	}
	public Set<Byte> getIndexesOfAll()
	{
		return map.keySet();
	}
	
/*	public String toString()
	{
		String str="";
		for(Entry<Byte,Integer> e: map.entrySet())
			str+=e.getKey()+" <--> "+cache.get(e.getValue())+"\n";
		return str;
	}*/
	
	public String toString()
	{
		String str="numValid: "+numValid+"\n"+map.toString()+"\n"+cache.toString();
		return str;
	}
	
}
