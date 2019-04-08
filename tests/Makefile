include Makefile.conf

INCDIR = ../include
SRCDIR = ../src
OBJDIR = ../obj

INCLUDE = -I$(INCDIR) -I$(EIGENDIR) -I$(BLASDIR) -I$(METISDIR) -I$(MMIODIR)

CFLAGS = -std=c++11 -Wall
ifdef DEBUG
    CFLAGS += -g -O0 -D_GLIBCXX_DEBUG
else
    CFLAGS += -O3 -DEIGEN_NO_DEBUG
endif
ifeq ($(USE_MKL),1)
    CFLAGS += -DUSE_MKL
endif

LDFLAGS = -lmetis -L$(METISLIB) -L$(BLASLIB) 
ifeq ($(USE_MKL),1)
    LDFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl 
else
    LDFLAGS += -lblas -llapack
endif

DEPS = $(INCDIR)/tree.h $(INCDIR)/util.h $(INCDIR)/is.h #$(INCDIR)/mm_wrap.h $(INCDIR)/mm_io.h
OBJ = $(OBJDIR)/tree.o $(OBJDIR)/util.o $(OBJDIR)/is.o  #$(OBJDIR)/mm_wrap.o $(OBJDIR)/mm_io.o

all: run icesheet mfem mfem2 mfem5 mfem9 tests spaND stokes

# Tests
$(OBJDIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(INCLUDE)

# Source
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(INCLUDE)

# Executables
tests: tests.cpp $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) -pthread libgtest.a -isystem $(GTEST)/include

mfem: mfem.cpp $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) -I$(MFEMDIR) -L$(MFEMLIB) -lmfem

mfem2: mfem2.cpp $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) -I$(MFEMDIR) -L$(MFEMLIB) -lmfem

mfem5: mfem5.cpp $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) -I$(MFEMDIR) -L$(MFEMLIB) -lmfem

mfem9: mfem9.cpp $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) -I$(MFEMDIR) -L$(MFEMLIB) -lmfem

spaND: spaND.cpp $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)

icesheet: icesheet.cpp $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)

stokes: stokes.cpp $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)

run: run.cpp $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)

libgtest.a: 
	cd $(GTEST); $(CC) -isystem ./include -I. -pthread -c ./src/gtest-all.cc; ar -rv ../../tests/libgtest.a gtest-all.o

.PHONY: clean

clean:
	rm -f $(OBJDIR)/*.o run icesheet spemfem mfem2 mfem9 tests spaND stokes

