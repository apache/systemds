package org.apache.sysds.hops.rewriter;

import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.rewrite.HopRewriteRule;
import org.apache.sysds.hops.rewrite.ProgramRewriteStatus;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class RewriteAutomaticallyGenerated extends HopRewriteRule {
	public static final String FILE_PATH_CONDITIONAL = "/Users/janniklindemann/Dev/MScThesis/rules_conditional.rl";
	public static final String FILE_PATH = "/Users/janniklindemann/Dev/MScThesis/rules.rl";
	public static final String VALIDATED_FILE_PATH = "/Users/janniklindemann/Dev/MScThesis/rules_validated.rl";
	public static final String RAW_FILE_PATH = "/Users/janniklindemann/Dev/MScThesis/raw_rules.rl";
	public static final String FILE_PATH_MB = "/Users/janniklindemann/Dev/MScThesis/rules_mb.rl";
	public static RewriteAutomaticallyGenerated existingRewrites;

	private Function<Hop, Hop> rewriteFn;

	public RewriteAutomaticallyGenerated() {
		// Try to read the file
		try {
			final RuleContext ctx = RewriterUtils.buildDefaultContext();
			List<String> lines = Files.readAllLines(Paths.get(FILE_PATH));
			RewriterRuleSet ruleSet = RewriterRuleSet.deserialize(lines, ctx);

			rewriteFn = ruleSet.compile("AutomaticallyGeneratedRewriteFunction", false);
			existingRewrites = this;
		} catch (IOException e) {
		}
	}

	public RewriteAutomaticallyGenerated(Function<Hop, Hop> rewriteFn) {
		this.rewriteFn = rewriteFn;
	}

	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if( roots == null || rewriteFn == null )
			return roots;

		/*System.out.println("Rewriting DAGs...");
		StackTraceElement[] ste = Thread.currentThread().getStackTrace();
		System.out.println("Caller:\t" + ste[2]);
		System.out.println("\t\t\t" + ste[3]);
		long startMillis = System.currentTimeMillis();*/

		//one pass rewrite-descend (rewrite created pattern)
		for( Hop h : roots )
			rule_apply( h, false );
		Hop.resetVisitStatus(roots, true);

		//one pass descend-rewrite (for rollup)
		for( Hop h : roots )
			rule_apply( h, true );

		//System.out.println("Generated rewrites took " + (System.currentTimeMillis() - startMillis) + "ms");

		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		if( root == null || rewriteFn == null )
			return root;

		//long startMillis = System.currentTimeMillis();

		//one pass rewrite-descend (rewrite created pattern)
		rule_apply( root, false );

		root.resetVisitStatus();

		//one pass descend-rewrite (for rollup)
		rule_apply( root, true );

		//System.out.println("Generated rewrites took " + (System.currentTimeMillis() - startMillis) + "ms");

		return root;
	}

	private void rule_apply(Hop hop, boolean descendFirst)
	{
		if(hop.isVisited())
			return;

		//DMLExecutor.println("Hop: " + hop + ", " + hop.getName() + ": " + HopRewriteUtils.isSparse(hop));
		//DMLExecutor.println("NNZ: " + hop.getNnz());

		//System.out.println("Stepping into: " + hop);

		//recursively process children
		for( int i=0; i<hop.getInput().size(); i++)
		{
			Hop hi = hop.getInput().get(i);

			//process childs recursively first (to allow roll-up)
			if( descendFirst )
				rule_apply(hi, descendFirst); //see below

			//apply actual simplification rewrites (of childs incl checks)
			hi = rewriteFn.apply(hi);

			//process childs recursively after rewrites (to investigate pattern newly created by rewrites)
			if( !descendFirst )
				rule_apply(hi, descendFirst);
		}

		hop.setVisited();
	}
}
