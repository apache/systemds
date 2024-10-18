package org.apache.sysds.hops.rewriter;

import org.apache.commons.collections4.bidimap.DualHashBidiMap;
import scala.Tuple2;

import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.function.Consumer;

public class RewriterFactTable {
	private HashMap<RewriterRule.IdentityRewriterStatement, Integer> factCounter = new HashMap<>();
	private DualHashBidiMap<RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement> relation = new DualHashBidiMap<>();
	private HashMap<Tuple2<RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement>, Integer> factTable = new HashMap<>();

	public boolean tryOrder(List<RewriterStatement> uncertainties) {
		try {
			// This is just to validate that an absolute order is possible
			RewriterUtils.forEachDistinctBinaryCombination(uncertainties, (id1, id2) -> {
				Integer comparison = tryCompare(new RewriterRule.IdentityRewriterStatement(id1), new RewriterRule.IdentityRewriterStatement(id2));

				if (comparison == null)
					throw new IllegalArgumentException();
			});
		} catch (IllegalArgumentException ex) {
			return false;
		}

		uncertainties.sort(this::compare);
		return true;
	}

	public void generateFact(List<RewriterStatement> uncertainties) {
		if (uncertainties.size() < 2)
			throw new IllegalArgumentException();

		RewriterRule.IdentityRewriterStatement stmt1 = new RewriterRule.IdentityRewriterStatement(uncertainties.stream().min(Comparator.comparingInt(this::numFacts)).get());
		RewriterRule.IdentityRewriterStatement maxOfStmt1 = max(stmt1, null);
		RewriterRule.IdentityRewriterStatement stmt2 = null;

		for (int i = 1; i < uncertainties.size(); i++) {
			stmt2 = new RewriterRule.IdentityRewriterStatement(uncertainties.get(i));

			if (stmt1.stmt != uncertainties.get(i) && tryCompare(stmt1, stmt2) == null)
				break;

			stmt2 = null;
		}

		if (stmt2 == null)
			throw new IllegalArgumentException();

		// As maxOfStmt1 is a maximum, we chain it to a minimum
		RewriterRule.IdentityRewriterStatement minOfStmt2 = min(stmt2, null);

		// TODO: Deal with cycles that could (maybe) be created
		relation.put(minOfStmt2, maxOfStmt1);

		factCounter.put(maxOfStmt1, factCounter.getOrDefault(maxOfStmt1, 0)+1);
		factCounter.put(minOfStmt2, factCounter.getOrDefault(minOfStmt2, 0)+1);

		List<RewriterRule.IdentityRewriterStatement> maxCluster = new ArrayList<>();
		maxCluster.add(minOfStmt2);
		max(minOfStmt2, el -> {
			maxCluster.add(el);
		});

		List<RewriterRule.IdentityRewriterStatement> minCluster = new ArrayList<>();
		minCluster.add(maxOfStmt1);
		min(maxOfStmt1, el -> {
			minCluster.add(el);
		});

		for (RewriterRule.IdentityRewriterStatement maxElement : maxCluster)
			for (RewriterRule.IdentityRewriterStatement minElement : minCluster)
				factTable.put(new Tuple2<>(maxElement, minElement), 1);
	}

	public Integer compare(RewriterStatement stmt1, RewriterStatement stmt2) {
		if (stmt1 == stmt2)
			return 0;

		return tryCompare(new RewriterRule.IdentityRewriterStatement(stmt1), new RewriterRule.IdentityRewriterStatement(stmt2));
	}

	private boolean isMin(RewriterRule.IdentityRewriterStatement stmt) {
		return relation.get(stmt) == null;
	}

	private boolean isMax(RewriterRule.IdentityRewriterStatement stmt) {
		return relation.getKey(stmt) == null;
	}

	private RewriterRule.IdentityRewriterStatement min(RewriterRule.IdentityRewriterStatement stmt, @Nullable Consumer<RewriterRule.IdentityRewriterStatement> consumer) {
		RewriterRule.IdentityRewriterStatement next = relation.get(stmt);

		while (next != null) {
			stmt = next;
			if (consumer != null)
				consumer.accept(stmt);
			next = relation.get(stmt);
		}

		return stmt;
	}

	private RewriterRule.IdentityRewriterStatement max(RewriterRule.IdentityRewriterStatement stmt, @Nullable Consumer<RewriterRule.IdentityRewriterStatement> consumer) {
		RewriterRule.IdentityRewriterStatement next = relation.getKey(stmt);

		while (next != null) {
			stmt = next;
			if (consumer != null)
				consumer.accept(stmt);
			next = relation.getKey(stmt);
		}

		return stmt;
	}

	private int numFacts(RewriterStatement stmt) {
		Integer num = factCounter.get(new RewriterRule.IdentityRewriterStatement(stmt));
		return num == null ? 0 : num;
	}

	private Integer tryCompare(RewriterRule.IdentityRewriterStatement id1, RewriterRule.IdentityRewriterStatement id2) {
		Integer result = factTable.get(new Tuple2<>(id1, id2));

		if (result != null)
			return result;

		result = factTable.get(new Tuple2<>(id2, id1));

		if (result == null)
			return null;

		return -result;
	}
}
