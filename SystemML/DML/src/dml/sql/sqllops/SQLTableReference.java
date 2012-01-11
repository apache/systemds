package dml.sql.sqllops;

public class SQLTableReference implements ISQLTableReference {
	
	public SQLTableReference()
	{
		
	}
	public SQLTableReference(String name, String alias)
	{
		this.name = name;
		this.alias = alias;
	}
	public SQLTableReference(String name)
	{
		this.name = name;
	}
	
	String name;
	String alias;
	
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public String getAlias() {
		return alias;
	}
	public void setAlias(String alias) {
		this.alias = alias;
	}
	
	public String toString()
	{
		String s = this.name;
		if(this.alias != null && this.alias.length() > 0)
			s += " " + alias;
		return s;
	}
}
