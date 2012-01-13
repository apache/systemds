package dml.sql.sqllops;

public class SQLUnion implements ISQLSelect {
	public enum UNIONTYPE
	{
		NONE,
		UNIONALL,
		UNION
	}
	
	public SQLUnion()
	{
		
	}
	
	public SQLUnion(ISQLSelect s1, ISQLSelect s2, UNIONTYPE type)
	{
		select1 = s1;
		select2 = s2;
		unionType = type;
	}
	
	ISQLSelect select1;
	ISQLSelect select2;
	UNIONTYPE unionType;
	
	public ISQLSelect getSelect1() {
		return select1;
	}
	public void setSelect1(ISQLSelect select1) {
		this.select1 = select1;
	}
	public ISQLSelect getSelect2() {
		return select2;
	}
	public void setSelect2(ISQLSelect select2) {
		this.select2 = select2;
	}
	public UNIONTYPE getUnionType() {
		return unionType;
	}
	public void setUnionType(UNIONTYPE unionType) {
		this.unionType = unionType;
	}
	
	public String toString()
	{
		String union = NEWLINE + "UNION" + NEWLINE;
		if(this.unionType == UNIONTYPE.UNIONALL)
			union = NEWLINE + "UNION ALL" + NEWLINE;
		
		return select1.toString() + union + select2.toString();
	}
}
