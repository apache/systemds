package org.apache.sysds.hops.rewriter;

import java.util.HashMap;
import java.util.Objects;

public class RewriterStatementLink {
	public RewriterStatement stmt;
	public int dagID;
	public HashMap<RewriterStatementLink, RewriterStatementLink> links;

	public RewriterStatementLink(final RewriterStatement stmt, final int dagID, final HashMap<RewriterStatementLink, RewriterStatementLink> links) {
		this.stmt = stmt;
		this.dagID = dagID;
		this.links = links;
	}

	@Override
	public int hashCode(){
		return Objects.hash(stmt, dagID);
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) return true;
		if (o == null || getClass() != o.getClass()) return false;
		RewriterStatementLink link = (RewriterStatementLink) o;
		return dagID == link.dagID && Objects.equals(stmt, link.stmt);
	}
}
