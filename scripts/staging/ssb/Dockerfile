# Help: https://docs.docker.com/compose/gettingstarted/#step-1-set-up

FROM postgres:latest
# Copy data into container
COPY data_dir tmp
# Init the data and load to the database with a sql script.
COPY other/ssb_init.sql /docker-entrypoint-initdb.d/
WORKDIR /tmp
RUN sed -i 's/|$//' "customer.tbl"
RUN sed -i 's/|$//' "part.tbl"
RUN sed -i 's/|$//' "supplier.tbl"
RUN sed -i 's/|$//' "date.tbl"
RUN sed -i 's/|$//' "lineorder.tbl"

