-- Use https://github.com/eyalroz/ssb-dbgen/blob/master/doc/ssb.ddl
-- A bit modified.
-- Drop tables if they exist
DROP TABLE IF EXISTS lineorder CASCADE;
DROP TABLE IF EXISTS customer CASCADE;
DROP TABLE IF EXISTS part CASCADE;
DROP TABLE IF EXISTS supplier CASCADE;
DROP TABLE IF EXISTS date CASCADE;

-- Date dimension
CREATE TABLE date (
    d_datekey            INTEGER NOT NULL,
    d_date               VARCHAR(19) NOT NULL,
    d_dayofweek          VARCHAR(10) NOT NULL,
    d_month              VARCHAR(10) NOT NULL,
    d_year               INTEGER NOT NULL,
    d_yearmonthnum       INTEGER NOT NULL,
    d_yearmonth          VARCHAR(8) NOT NULL,
    d_daynuminweek       INTEGER NOT NULL,
    d_daynuminmonth      INTEGER NOT NULL,
    d_daynuminyear       INTEGER NOT NULL,
    d_monthnuminyear     INTEGER NOT NULL,
    d_weeknuminyear      INTEGER NOT NULL,
    d_sellingseason      VARCHAR(13) NOT NULL,
    d_lastdayinweekfl    VARCHAR(1) NOT NULL,
    d_lastdayinmonthfl   VARCHAR(1) NOT NULL,
    d_holidayfl          VARCHAR(1) NOT NULL,
    d_weekdayfl          VARCHAR(1) NOT NULL
);

-- Customer dimension
CREATE TABLE customer 
(
    c_custkey      INTEGER NOT NULL,
    c_name         VARCHAR(25) NOT NULL,
    c_address      VARCHAR(25) NOT NULL,
    c_city         VARCHAR(10) NOT NULL,
    c_nation       VARCHAR(15) NOT NULL,
    c_region       VARCHAR(12) NOT NULL,
    c_phone        VARCHAR(15) NOT NULL,
    c_mktsegment   VARCHAR(10) NOT NULL
);

-- Part dimension
CREATE TABLE part (
    p_partkey     INTEGER NOT NULL,
    p_name        VARCHAR(22) NOT NULL,
    p_mfgr        VARCHAR(6),
    p_category    VARCHAR(7) NOT NULL,
    p_brand      VARCHAR(9) NOT NULL,
    p_color       VARCHAR(11) NOT NULL,
    p_type        VARCHAR(25) NOT NULL,
    p_size        INTEGER NOT NULL,
    p_container   VARCHAR(10) NOT NULL
);

-- Supplier dimension
CREATE TABLE supplier (
    s_suppkey   INTEGER NOT NULL,
    s_name      VARCHAR(25) NOT NULL,
    s_address   VARCHAR(25) NOT NULL,
    s_city      VARCHAR(10) NOT NULL,
    s_nation    VARCHAR(15) NOT NULL,
    s_region    VARCHAR(12) NOT NULL,
    s_phone     VARCHAR(15) NOT NULL
);

-- LineOrder fact table
CREATE TABLE lineorder (
    lo_orderkey          INTEGER NOT NULL,
    lo_linenumber        INTEGER NOT NULL,
    lo_custkey           INTEGER NOT NULL,
    lo_partkey           INTEGER NOT NULL,
    lo_suppkey           INTEGER NOT NULL,
    lo_orderdate         INTEGER NOT NULL,
    lo_orderpriority     VARCHAR(15) NOT NULL,
    lo_shippriority      VARCHAR(1) NOT NULL,
    lo_quantity          INTEGER NOT NULL,
    lo_extendedprice     INTEGER NOT NULL,
    lo_ordertotalprice   INTEGER NOT NULL,
    lo_discount          INTEGER NOT NULL,
    lo_revenue           INTEGER NOT NULL,
    lo_supplycost        INTEGER NOT NULL,
    lo_tax               INTEGER NOT NULL,
    lo_commitdate        INTEGER NOT NULL,
    lo_shipmode          VARCHAR(10) NOT NULL
);

ALTER TABLE date
ADD PRIMARY KEY(d_datekey);

ALTER TABLE supplier
ADD PRIMARY KEY(s_suppkey);

ALTER TABLE customer 
ADD PRIMARY KEY (c_custkey);

ALTER TABLE part
ADD PRIMARY KEY (p_partkey);

--ALTER TABLE lineorder
--ADD PRIMARY KEY (lo_orderkey);

--ALTER TABLE lineorder
--ADD FOREIGN KEY (lo_orderdate) REFERENCES date (d_datekey);

--ALTER TABLE lineorder
--ADD FOREIGN KEY (lo_commitdate) REFERENCES date (d_datekey);

--ALTER TABLE lineorder
--ADD FOREIGN KEY (lo_suppkey) REFERENCES supplier (s_suppkey);

--ALTER TABLE lineorder
--ADD FOREIGN KEY (lo_custkey) REFERENCES customer (c_custkey);

--ALTER TABLE lineorder
--ADD FOREIGN KEY (lo_partkey) REFERENCES part (p_partkey);

-- Copying data inside.
COPY customer FROM '/tmp/customer.tbl' DELIMITER '|';
COPY part FROM '/tmp/part.tbl' DELIMITER '|';
COPY supplier FROM '/tmp/supplier.tbl' DELIMITER '|';
COPY date FROM '/tmp/date.tbl' DELIMITER '|';
COPY lineorder FROM '/tmp/lineorder.tbl' DELIMITER '|';