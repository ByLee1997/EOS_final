CXX= arm-unknown-linux-gnu-g++
OBJS= creator_lib.o sockop.o
CFLAG= -g -Wall -O0
# CFLAG=-gdwarf-2 -O0
TARGETS= cmos_test client_plate

COLOR_YELLOW='\033[0;33m'
COLOR_NC='\033[0m'

INCLUDE1 = /home/samliu/Documents/eos/lab1/microtime/linux/include/
INCLUDE2 = /opt/microtime/pro/devkit/arm/pxa270/gcc-4.0.2-glibc-2.3.3/arm-unknown-linux-gnu/include/
LINK = /opt/microtime/pro/devkit/arm/pxa270/gcc-4.0.2-glibc-2.3.3/arm-unknown-linux-gnu/lib/
 
all: $(TARGETS)
	@echo -e ${COLOR_YELLOW}'Remember to check whether the [INCLUDE1, INCLUDE2, LINK] variables are correct or not in Makefile.'${COLOR_NC}

# Dependencies
%.o: %.c
	$(CXX) -c $< -o $@  -L $(LINK) -I $(INCLUDE1) -I $(INCLUDE2) $(CFLAG)
	
# My cpp files
%.o: %.cpp
	$(CXX) -c $< -o $@  -L $(LINK) -I $(INCLUDE1) -I $(INCLUDE2) $(CFLAG)

cmos_test: $(OBJS) cmos_test.o
	$(CXX) -o $@ $^ -L $(LINK) -I $(INCLUDE1) -I $(INCLUDE2) $(CFLAG) -pthread

client_plate: $(OBJS) client_plate.o
	$(CXX) -o $@ $^ -L $(LINK) -I $(INCLUDE1) -I $(INCLUDE2) $(CFLAG) -pthread

clean:
	rm -f *.o $(TARGETS) $(OBJS)

.PHONY: all clean
