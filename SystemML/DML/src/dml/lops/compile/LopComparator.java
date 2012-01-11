package dml.lops.compile;

import java.util.Comparator;

import dml.lops.KEY;
import dml.lops.Lops;
import dml.lops.VAL;
import dml.utils.DMLRuntimeException;

/**
 * 
 * Comparator class used in sorting the LopDAG in topological order. Refer to
 * doTopologicalSort_strict_order() in dml/lops/compile/Dag.java
 * 
 * Topological sort guarantees the following:
 * 
 * 1) All lops with level i appear before any lop with level greater than i
 * (source nodes are at level 0)
 * 
 * 2) Within a given level, nodes are ordered by their ID i.e., by the other in
 * which they are created
 * 
 * compare() method is designed to respect the above two requirements.
 * 
 * @author Shirish
 * 
 * @param <N>
 */
public class LopComparator<N extends Lops>
		implements Comparator<N> {
	@Override
	public int compare(N o1, N o2) {
		if (o1.getLevel() < o2.getLevel())
			return -1; // o1 is less than o2
		else if (o1.getLevel() > o2.getLevel())
			return 1; // o1 is greater than o2
		else {
			if (o1.getID() < o2.getID())
				return -1; // o1 is less than o2
			else if (o1.getID() > o2.getID())
				return 1; // o1 is greater than o2
			else
				try {
					throw new DMLRuntimeException(
							"Unexpected error: ID's of two lops are same.");
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
		}

		return 0; // should never be reached
	}

}
