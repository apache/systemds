package com.ibm.bi.dml.sql.sqlcontrolprogram;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.HashMap;

import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;

/*
 * BIRelease: Following code is commented for BigInsights Release. 
 */
// import org.netezza.error.NzSQLException;


public class NetezzaConnector {
	
	public NetezzaConnector()
	{
		
	}
	
	public NetezzaConnector(String server, String dbName, String schema, String user, String password)
	{
		this.server = server;
		this.dbName = dbName;
		this.schema = schema;
		this.user = user;
		this.pwd = password;
	}
	
	Connection conn;
	boolean connected = false;
	String server = "12.36.6.222";
    Integer port = 5480;
    String dbName = "systemmldb";
    String user = "sysmluser";
    String pwd = "sysmlpassword";
    String schema = "schema";
    String driverClass = "org.netezza.Driver";

    public static final String VARTABLENAME = "SystemML_Variables";

    private String getURL()
    {
    	return "jdbc:netezza://" + server + ":" + port.toString() + "/" + dbName ;
    }
    
    public void connect(boolean autocommit) throws SQLException, ClassNotFoundException
    {
    	connect();
    	conn.setAutoCommit(autocommit);
    }
    
    public void connect() throws SQLException, ClassNotFoundException
    {
    	Class.forName(driverClass);
        conn = DriverManager.getConnection(getURL(), user, pwd);
        connected = true;
    }
    
    public void disconnect() throws SQLException
    {
    	if(conn != null)
    		conn.close();
    	connected = false;
    }
    
    /**
     * Executes SQL code on the database
     * @param sql The SQL code to be executed
     * @param isQuery If true, a result set is returned
     * @return Returns a result set, if isQuery is true
     * @throws SQLException 
     */
    public void executeSQL(String sql) throws SQLException
    {
    	checkConnected();
    	Statement st = null;
    	/*
    	 * BIRelease: Following code is commented for BigInsights Release. 
    	 */
//        try {

            st = conn.createStatement();
            st.execute(sql);
//        }
/*        catch(NzSQLException e)
        {
        	if(e.getErrorCode() == 1012 && e.getMessage().contains("Integer.MAX_VALUE"))
        		System.out.println("WARNING: " + e.getErrorCode() + " " + e.getMessage() + "\r\n" + e.getLocalizedMessage());
        	else throw e;
        }
        finally {
            try {
                if( st!= null)
                    st.close();
            } catch (SQLException e1) {
                    e1.printStackTrace();
                }
        }
*/    }
    
    /**
     * Calls a procedure without parameters as created by SystemML on the database
     * @param name The name of the procedure
     * @throws SQLException 
     */
    public void callProcedure(String name) throws SQLException
    {
    	checkConnected();
    	executeSQL("call " + name + "();");
    }
    
    public String getScalarString(String query) throws SQLException
    {
    	checkConnected();
    	Statement st = null;
    	ResultSet rs = null;
    	String res = null;
        try {

            st = conn.createStatement();
            rs = st.executeQuery(query);
            
            if(rs.next())
            {
            	res = rs.getString(1);
            }
            
        }  finally {
            try {
                if( st!= null)
                    st.close();
            } catch (SQLException e1) {
                    e1.printStackTrace();
                }
        }
        return res;
    }
    
    public double getScalarDouble(String query) throws SQLException
    {
    	checkConnected();
    	Statement st = null;
    	ResultSet rs = null;
    	double res = 0.0;
        try {

            st = conn.createStatement();
            rs = st.executeQuery(query);
            
            if(rs.next())
            {
            	res = rs.getDouble(1);
            	//double meta = rs.getInt(2);
            	//TODO Check meta and change res to +/-Infinity or NaN accordingly
            }
            else
            	throw new SQLException("ERROR: No result set was returned");
            
        } finally {
            try {
                if( st!= null)
                    st.close();
            } catch (SQLException e1) {
                    e1.printStackTrace();
                }
        }
        return res;
    }
    
    public boolean getScalarBoolean(String query) throws SQLException
    {
    	checkConnected();
    	Statement st = null;
    	ResultSet rs = null;
    	boolean res = false;
        try {
            st = conn.createStatement();
            rs = st.executeQuery(query);
            
            if(rs.next())
            {
            	res = rs.getBoolean(1);
            }
            else
            	throw new SQLException("ERROR: No result set was returned");
            
        }  finally {
            try {
                if( st!= null)
                    st.close();
            } catch (SQLException e1) {
                    e1.printStackTrace();
                }
        }
        return res;
    }
    
    public int getScalarInteger(String query) throws SQLException
    {
    	checkConnected();
    	Statement st = null;
    	ResultSet rs = null;
    	int res = 0;
        try {

            st = conn.createStatement();
            rs = st.executeQuery(query);
            
            if(rs.next())
            {
            	res = rs.getInt(1);
            }
            else
            	throw new SQLException("ERROR: No result set was returned");
            
        } finally {
            try {
                if( st!= null)
                    st.close();
            } catch (SQLException e1) {
                    e1.printStackTrace();
                }
        }
        return res;
    }
    
    /**
     * Dumps a table that contains a matrix to a file
     * @param tableName
     * @param fileName
     * @throws SQLException
     * @throws IOException
     */
    public void tableToFile(String tableName, String fileName) throws SQLException, IOException
    {
    	checkConnected();
    	
    	ResultSet rs = null;
    	Statement st = null;

        try {
            st = conn.createStatement();
            rs = st.executeQuery(String.format("SELECT * FROM \"%s\"", tableName));
            
            BufferedWriter bw = null;
        	try
        	{
    	    	File file = new File(fileName);
    	    	bw = new BufferedWriter(new FileWriter(file));
    	    	while(rs.next())
    	    	{
    	    		String row = String.format("%s %s %s\r\n", rs.getString("row"), rs.getString("col"), rs.getString("value"));
    	    		bw.write(row);
    	    	}
        	}
        	finally
        	{
        		if(bw != null)
        			bw.close();
        	}

        }  finally {
            try {
            	if(rs != null)
            		rs.close();
                if( st!= null)
                    st.close();
            } catch (SQLException e1) {
                    e1.printStackTrace();
                }
        }
    }
    
    /**
     * Returns a table that contains a matrix as a HashMap
     * @param tableName
     * @return
     * @throws SQLException
     */
    public HashMap<CellIndex,Double> tableToHashMap(String tableName) throws SQLException
    {
    	checkConnected();
    	
    	ResultSet rs = null;
    	Statement st = null;
    	HashMap<CellIndex, Double> output = new HashMap<CellIndex, Double>();
        try {
            st = conn.createStatement();
            rs = st.executeQuery(String.format("SELECT * FROM \"%s\"", tableName));

	    	while(rs.next())
	    	{
	    		output.put(new CellIndex(rs.getInt("row"), rs.getInt("col")), rs.getDouble("value"));
	    	}
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
            	if(rs != null)
            		rs.close();
                if( st!= null)
                    st.close();
            } catch (SQLException e1) {
                    e1.printStackTrace();
                }
        }
        return output;
    }
    
    /**
     * Creates a matrix table from a file with matrix data
     * @param tableName
     * @param fileName
     * @throws SQLException
     * @throws IOException
     */
    public void createTable(String tableName, String fileName, boolean firstIsMetaData) throws SQLException, IOException
    {
    	checkConnected();
    	
    	BufferedReader reader = null;
    	Statement st = null;

        try {
            st = conn.createStatement();
            st.execute("call drop_if_exists('" + tableName + "'); ");
            st.execute("CREATE TABLE \"" + tableName + "\" (row int8, col int8, value int8); ");
            
            File file = new File(fileName);
            reader = new BufferedReader(new FileReader(file));
            
            String line = reader.readLine();
            if(firstIsMetaData)
            {
            	reader.readLine();
            	line = reader.readLine();
            }
            StringBuffer query = new StringBuffer();
            while(line != null)
            {
	            String[] split = line.split(" ");
	            if(split.length == 3)
	            {
		        	query.append("INSERT INTO ");
		        	query.append("\"" + tableName + "\"");
		        	query.append(" VALUES (");
		        	query.append(split[0]);
		        	query.append(", ");
		        	query.append(split[1]);
		        	query.append(", ");
		        	query.append(split[2]);
		        	query.append("); ");
		        	System.out.println(split[0] + "," + split[1] + ": " + split[2]);
	            }
	            line = reader.readLine();
            	String sql = query.toString();
            	st.addBatch(sql);
                //Reset StringBuffer
                query.setLength(0);
            }
            st.executeBatch();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
            	if(reader != null)
            		reader.close();
                if( st!= null)
                    st.close();
                if( conn != null)
                    conn.close();
            } catch (SQLException e1) {
                    e1.printStackTrace();
                }
        }
    }
    
    public void exportHadoopDirectoryToNetezza(String dirName, String tablename, boolean createTable) throws SQLException, IOException
    {
    	checkConnected();
    	
    	File dir = new File(dirName);
    	Statement st = conn.createStatement();
    	String template = 
        	"INSERT INTO \"%s\" SELECT * FROM EXTERNAL '%s' USING ( DELIMITER ' ' Y2BASE 2000 ENCODING 'internal' REMOTESOURCE 'JDBC' ESCAPECHAR '\')";
        	
    	try
    	{
	    	if(createTable)
	    	{
	            st.execute("call drop_if_exists('" + tablename + "'); ");
	            st.execute("CREATE TABLE \"" + tablename + "\" (row int8, col int8, value double precision); ");
	    	}
	    	String[] files = dir.list();
	    	if(files != null)
	    	{
		    	for(String f : files)
		    	{
		    		File fi = new File(f);
		    		if(!fi.isDirectory() && !fi.getName().endsWith(".crc"))
		    		{
		    			String complete = String.format(template, tablename, dirName + "/" + fi.getName());
		                st.execute(complete);
		    		}
		    	}
	    	}
    	}
    	catch(Exception e) {
        	e.printStackTrace();
        } finally {
        try {
            if( st!= null)
                st.close();
        } catch (SQLException e1) {
                e1.printStackTrace();
            }
        }
    }
    
    public void exportTable(String tableName, String filename) throws SQLException
    {
    	checkConnected();
    	String template = 
    	"INSERT INTO \"%s\" SELECT * FROM EXTERNAL '%s' USING ( DELIMITER ' ' Y2BASE 2000 ENCODING 'internal' REMOTESOURCE 'JDBC' ESCAPECHAR '\')";
    	Statement st = null;

        try {
            st = conn.createStatement();
            st.execute("call drop_if_exists('" + tableName + "'); ");
            st.execute("CREATE TABLE \"" + tableName + "\" (row int8, col int8, value double precision) DISTRIBUTE ON (row, col); ");
            String complete = String.format(template, tableName, filename);
            st.execute(complete);
        } catch (Exception e) {
        	e.printStackTrace();
        } finally {
        try {
            if( st!= null)
                st.close();
        } catch (SQLException e1) {
                e1.printStackTrace();
            }
        }
    }
    
    /**
     * Creates a matrix table with data from a two dimensional array
     * @param tableName
     * @param data
     * @throws SQLException
     */
    public void createTable(String tableName, double[][] data) throws SQLException
    {
    	checkConnected();
    	
    	Statement st = null;
        try {
            st = conn.createStatement();

            int rows = data.length;
            int cols = data[0].length;
            st.execute("call drop_if_exists('" + tableName + "'); ");
            st.execute("CREATE TABLE \"" + tableName + "\" (row int8, col int8, value double precision); ");
            //String tmpl = "INSERT INTO \"%s\" VALUES (%d,%d,%f);";
            conn.setAutoCommit(false);
            for(int x = 1; x <= cols; x++)
            {
            	for(int y = 1; y <= rows; y++)
                {
            		if(data[y-1][x-1] == 0)
            			continue;
            		String sql = "INSERT INTO \"" + tableName + "\" VALUES (" + y + "," + x + "," + data[y-1][x-1] + ");";
            		//String sql = String.format(new Locale("en-US"), tmpl, tableName,y,x,data[y-1][x-1]);
	                st.addBatch(sql);
                }
            }
            st.executeBatch();
            conn.commit();
            conn.setAutoCommit(true);
            
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if( st!= null)
                    st.close();
            } catch (SQLException e1) {
                    e1.printStackTrace();
                }
        }
    }
    
    public void setVariableValue(String name, Object value) throws SQLException
    {
    	checkConnected();
    	
    	String valueString = value.toString();
    	String updateSQL = String.format("UPDATE %s SET sval = %s WHERE name = %s", VARTABLENAME, valueString, name);
    	
    	Statement st = null;
    	try {
            st = conn.createStatement();
            st.execute(updateSQL);
    	}
    	catch(Exception e)
    	{
    		e.printStackTrace();
    	}
    	finally
    	{
    		if(st != null)
    			st.close();
    	}
    }
    
    public <T> T getVariableValue(String varName) throws SQLException
    {
    	checkConnected();
    	
    	Statement st = null;
    	ResultSet rs = null;
    	Object output = null;
    	
        try {
            st = conn.createStatement();
            String query = String.format("SELECT sval FROM %s WHERE name = %s", VARTABLENAME,varName);
            rs = st.executeQuery(query);
            
            if(rs.next())
            	output = rs.getObject(1);
            
        } catch (Exception e) {
        	e.printStackTrace();
	    } finally {
	        try {
	            if( st!= null)
	                st.close();
	            if(rs != null)
	            	rs.close();
	        	} catch (SQLException e1) {
	                e1.printStackTrace();
	            }
	    }
	    
	    return (T)output;
    }
    
    private void checkConnected() throws SQLException
    {
    	if(!connected)
    		throw new SQLException("ERROR: There is no connection to the database");
    }
    
	public String getServer() {
		return server;
	}

	public void setServer(String server) {
		this.server = server;
	}

	public String getDbName() {
		return dbName;
	}

	public void setDbName(String dbName) {
		this.dbName = dbName;
	}

	public String getUser() {
		return user;
	}

	public void setUser(String user) {
		this.user = user;
	}

	public String getPassword() {
		return pwd;
	}

	public void setPassword(String pwd) {
		this.pwd = pwd;
	}

	public String getSchema() {
		return schema;
	}

	public void setSchema(String schema) {
		this.schema = schema;
	}

	public String getDriverClass() {
		return driverClass;
	}

	public void setDriverClass(String driverClass) {
		this.driverClass = driverClass;
	}
	
	public Connection getConnection() {
		return conn;
	}

	public void finalize()
	{
		try
		{
			if(connected && conn != null)
				conn.close();
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}
}
