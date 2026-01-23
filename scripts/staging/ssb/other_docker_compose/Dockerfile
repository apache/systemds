# Help: https://docs.docker.com/compose/gettingstarted/#step-1-set-up
# https://docs.docker.com/build/building/multi-stage/
# Star Schema Benchmark data set generator (ssb-dbgen): 
# https://github.com/eyalroz/ssb-dbgen.git

# We use multi-stage method.

# First create the datagen docker container
FROM alpine:latest AS datagen
ARG SCALE
RUN echo "build and generate data with datagen with scale factor $SCALE"

RUN apk update
RUN apk add git gcc cmake make musl-dev
RUN git clone https://github.com/eyalroz/ssb-dbgen.git --depth 1
# Build the generator
WORKDIR /ssb-dbgen
RUN echo "build and generate data with datagen with scale factor $SCALE"
RUN cmake -B ./build && cmake --build ./build
# Run the generator (with -s )
RUN build/dbgen -b dists.dss -v -s $SCALE

# Second: use the systemds docker container
# And execute a selected query. 
FROM apache/systemds:latest
ENV QUERY q1_1
WORKDIR /systemds
COPY queries queries
#COPY --from=datagen /ssb-dbgen/data data 
COPY --from=datagen ssb-dbgen/customer.tbl tmp/customer.tbl
COPY --from=datagen ssb-dbgen/part.tbl data/part.tbl
COPY --from=datagen ssb-dbgen/supplier.tbl data/supplier.tbl
COPY --from=datagen ssb-dbgen/date.tbl data/date.tbl
COPY --from=datagen ssb-dbgen/lineorder.tbl data/lineorder.tbl
CMD ["queries/$QUERY.dml","systemds/data"]

#TODO: Currently only accepting one query each. To expand to accept more queries.
#COPY helper.sh helper.sh
#RUN chmod u+x helper.sh
# COPY host_data_dir container_data_dir
#COPY data/very_small_s0_01_dataset data

#CMD if [ "$QUERY" = "all" ]; then \
#      ["./helper.sh", "A", "B"]; \
#    else \
#      ["queries/$QUERY.dml","data"]; \
#    fi 

FROM postgres:latest
COPY ssb_init.sql /docker-entrypoint-initdb.d/
ENV QUERY q1_1
COPY --from=datagen ssb-dbgen/customer.tbl tmp/customer.tbl
COPY --from=datagen ssb-dbgen/part.tbl tmp/part.tbl
COPY --from=datagen ssb-dbgen/supplier.tbl tmp/supplier.tbl
COPY --from=datagen ssb-dbgen/date.tbl tmp/date.tbl
COPY --from=datagen ssb-dbgen/lineorder.tbl tmp/lineorder.tbl
WORKDIR /tmp
RUN sed -i 's/|$//' "customer.tbl"
RUN sed -i 's/|$//' "part.tbl"
RUN sed -i 's/|$//' "supplier.tbl"
RUN sed -i 's/|$//' "date.tbl"
RUN sed -i 's/|$//' "lineorder.tbl"
RUN cd .. && mkdir -p queries
WORKDIR /
COPY sql queries
WORKDIR /queries
RUN mv q1.1.sql q1_1.sql
RUN mv q1.2.sql q1_2.sql
RUN mv q1.3.sql q1_3.sql
RUN mv q2.1.sql q2_1.sql
RUN mv q2.2.sql q2_2.sql
RUN mv q2.3.sql q2_3.sql
RUN mv q3.1.sql q3_1.sql
RUN mv q3.2.sql q3_2.sql
RUN mv q3.3.sql q3_3.sql
RUN mv q3.4.sql q3_4.sql
RUN mv q4.1.sql q4_1.sql
RUN mv q4.2.sql q4_2.sql
RUN mv q4.3.sql q4_3.sql

