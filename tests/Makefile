include Makefile.conf

INCDIR = ../include
SRCDIR = ../src
OBJDIR = ../obj
BUILDDIR = ../build

## Include folder
INCLUDE = -I$(INCDIR) $(EXTRA_INCDIR)

## Link folder
LINK = $(EXTRA_LINKDIR)

## Compiler flags
CFLAGS = -std=c++14 -Wall -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE
ifdef DEBUG
    CFLAGS += -g -O0 -D_GLIBCXX_DEBUG
else
    CFLAGS += -O3 -DEIGEN_NO_DEBUG
endif
ifeq ($(USE_MKL),1)
    CFLAGS += -DUSE_MKL -DEIGEN_USE_MKL_ALL
endif
CFLAGS += $(EXTRA_CFLAGS)

## Linker flags
LDFLAGS = -lmetis
ifeq ($(USE_MKL),1)
    LDFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl 
else
    LDFLAGS += -lblas -llapack
endif

DEPS = $(INCDIR)/tree.h $(INCDIR)/util.h $(INCDIR)/is.h $(INCDIR)/partition.h $(INCDIR)/operations.h $(INCDIR)/cluster.h $(INCDIR)/edge.h $(INCDIR)/spaND.h
OBJ  = $(OBJDIR)/tree.o $(OBJDIR)/util.o $(OBJDIR)/is.o $(OBJDIR)/partition.o $(OBJDIR)/operations.o $(OBJDIR)/cluster.o $(OBJDIR)/edge.o

all: icesheet mfem mfem2 mfem5 mfem9 tests spaND

# Source
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(INCLUDE)

# Executables
tests: tests.cpp $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LINK) -pthread libgtest.a -isystem $(GTESTDIR)

mfem: mfem.cpp $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LINK) -I$(MFEMDIR) -L$(MFEMLIB) -lmfem

mfem2: mfem2.cpp $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LINK) -I$(MFEMDIR) -L$(MFEMLIB) -lmfem

mfem5: mfem5.cpp $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LINK) -I$(MFEMDIR) -L$(MFEMLIB) -lmfem

mfem9: mfem9.cpp $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LINK) -I$(MFEMDIR) -L$(MFEMLIB) -lmfem

spaND: spaND.cpp $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LINK) 

icesheet: icesheet.cpp $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LINK) 

.PHONY: clean libspaND

clean:
	rm -f $(OBJDIR)/*.o icesheet spemfem mfem2 mfem9 tests spaND

libspaND:
	cp $(INCDIR)/tree.h $(INCDIR)/util.h $(INCDIR)/is.h $(INCDIR)/partition.h $(INCDIR)/operations.h $(INCDIR)/cluster.h $(INCDIR)/edge.h $(INCDIR)/spaND.h $(BUILDDIR)/include/
	ar -rv $(BUILDDIR)/libspaND.a $(OBJDIR)/tree.o $(OBJDIR)/util.o $(OBJDIR)/is.o $(OBJDIR)/partition.o $(OBJDIR)/operations.o $(OBJDIR)/cluster.o $(OBJDIR)/edge.o

