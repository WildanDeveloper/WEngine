CC = g++
CXXFLAGS = -O3 -march=native -mavx512f -mavx512vl -mavx512bw -mavx512vnni -fPIC -Wall -Wextra -shared

all: libwildandev_accel.so

libwildandev_accel.so: wildandev_accel.cpp
	$(CC) $(CXXFLAGS) -o $@ $<

clean:
	rm -f libwildandev_accel.so

.PHONY: all clean
