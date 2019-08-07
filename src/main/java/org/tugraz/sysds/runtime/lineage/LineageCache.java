package org.tugraz.sysds.runtime.lineage;

import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.instructions.cp.ComputationCPInstruction;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;

import java.util.HashMap;
import java.util.Map;

public class LineageCache {
	private static final Map<LineageItem, MatrixBlock> _cache = new HashMap<>();
	
	public static void put(Instruction inst, ExecutionContext ec) {
		if (!DMLScript.LINEAGE_REUSE)
			return;
		
		if( inst instanceof ComputationCPInstruction
			&&((ComputationCPInstruction) inst).output.getDataType().isMatrix() ) {
			
			for (LineageItem item : ((LineageTraceable) inst).getLineageItems(ec)) {
				MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction) inst).output);
				LineageCache._cache.put(item, mo.acquireReadAndRelease());
			}
		}
	}
	
	public static boolean probe(LineageItem key) {
		return _cache.containsKey(key);
	}
	
	public static MatrixBlock get(LineageItem key) {
		return _cache.get(key);
	}
	
	public static void resetCache() {
		_cache.clear();
	}
	
	public static boolean reuse(Instruction inst, ExecutionContext ec) {
		if (!DMLScript.LINEAGE && DMLScript.LINEAGE_REUSE)
			return false;
		
		if (inst instanceof ComputationCPInstruction) {
			boolean reused = true;
			LineageItem[] items = ((ComputationCPInstruction) inst).getLineageItems(ec);
			for (LineageItem item : items) {
				if (LineageCache.probe(item))
					ec.setMatrixOutput(((ComputationCPInstruction) inst).output.getName(), LineageCache.get(item));
				else
					reused = false;
			}
			return reused && items.length > 0;
		} else {
			return false;
		}
	}
}
