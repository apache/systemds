# Help: https://docs.docker.com/compose/gettingstarted/#step-1-set-up
# Star Schema Benchmark data set generator (ssb-dbgen): 
# https://github.com/eyalroz/ssb-dbgen.git

FROM alpine:latest
WORKDIR /ssb-dbgen
ARG scale #scale-factor of the generated data.
RUN apk update
RUN apk add git gcc cmake make musl-dev
RUN git clone https://github.com/eyalroz/ssb-dbgen.git --depth 1
# Build the generator
WORKDIR /ssb-dbgen/ssb-dbgen
RUN echo "build and generate data with datagen."
RUN cmake -B ./build && cmake --build ./build
# Run the generator (with -s )
RUN build/dbgen -b dists.dss -v -s $scale
