package dml.runtime.matrix.mapred;

import java.util.ArrayList;

import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue;

public class CachedValueMap extends CachedMap<IndexedMatrixValue>{

	public IndexedMatrixValue set(byte thisMatrix, MatrixIndexes indexes, MatrixValue value) {
		if(numValid<cache.size())	
			cache.get(numValid).set(indexes, value);
		else
			cache.add(new IndexedMatrixValue(indexes, value));
		
		ArrayList<Integer> list=map.get(thisMatrix);
		if(list==null)
		{
			list=new ArrayList<Integer>(4);
			map.put(thisMatrix, list);
		}
		list.add(numValid);
		numValid++;
		return cache.get(numValid-1);
		
	}

	public IndexedMatrixValue holdPlace(byte thisMatrix, Class<? extends MatrixValue> cls)
	{
		if(numValid>=cache.size())	
			cache.add(new IndexedMatrixValue(cls));
		
		ArrayList<Integer> list=map.get(thisMatrix);
		if(list==null)
		{
			list=new ArrayList<Integer>(4);
			map.put(thisMatrix, list);
		}
		list.add(numValid);
		numValid++;
		return cache.get(numValid-1);
	}
}
