#
#  USAGE:
#     make          ... to build the program
#     make test     ... to run the default test case
#

# verify that you are including the right make.def file for the platform
include make.def

EXES     = mm_testbed$(EXE) cmass$(EXE) gauss_solv$(EXE) stencil$(EXE) jac_solv$(EXE) 

MM_OBJS  = mm_testbed.$(OBJ) mm_utils.$(OBJ) mm_ijk.$(OBJ) mm_tst_cases.$(OBJ)

CM_OBJS  = center_of_mass.$(OBJ) random.$(OBJ)

STN_OBJS  = stencil.$(OBJ)  

GAUS_OBJS  = gauss_solv.$(OBJ) mm_utils.$(OBJ) 

JAC_OBJS  = jac_solv.$(OBJ) mm_utils.$(OBJ) 

all: $(EXES)

stencil$(EXE): $(STN_OBJS) 
	$(CLINKER) $(CFLAGS) -o stencil$(EXE) $(STN_OBJS) $(LIBS)

jac_solv$(EXE): $(JAC_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o jac_solv$(EXE) $(JAC_OBJS) $(LIBS)

gauss_solv$(EXE): $(GAUS_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o gauss_solv$(EXE) $(GAUS_OBJS) $(LIBS)

mm_testbed$(EXE): $(MM_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o mm_testbed$(EXE) $(MM_OBJS) $(LIBS)

cmass$(EXE): $(CM_OBJS) random.h
	$(CLINKER) $(CFLAGS) -o cmass$(EXE) $(CM_OBJS) $(LIBS)

test: $(EXES)
	for i in $(EXES); do \
            $(PRE)$$i; \
        done

clean:
	$(RM) $(EXES) *.$(OBJ)

.SUFFIXES:
.SUFFIXES: .c .cpp .$(OBJ)

.c.$(OBJ):
	$(CC) $(CFLAGS) -c $<

.cpp.$(OBJ):
	$(CC) $(CFLAGS) -c $<


