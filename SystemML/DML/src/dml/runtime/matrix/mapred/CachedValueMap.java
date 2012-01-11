package dml.runtime.matrix.mapred;

import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue;

public class CachedValueMap extends CachedMap<IndexedMatrixValue>{

	public IndexedMatrixValue set(byte thisMatrix, MatrixIndexes indexes, MatrixValue value) {
		if(numValid<cache.size())	
			cache.elementAt(numValid).set(indexes, value);
		else
			cache.add(new IndexedMatrixValue(indexes, value));
		map.put(thisMatrix, numValid);
		numValid++;
		return cache.elementAt(numValid-1);
		
	}

	public IndexedMatrixValue holdPlace(byte thisMatrix, Class<? extends MatrixValue> cls)
	{
		if(numValid>=cache.size())	
			cache.add(new IndexedMatrixValue(cls));
		map.put(thisMatrix, numValid);
		numValid++;
		return cache.elementAt(numValid-1);
	}
}
