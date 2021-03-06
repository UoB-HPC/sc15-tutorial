#
#  USAGE:
#     make          ... to build the program
#     make test     ... to run the default test case
#

# verify that you are including the right make.def file for the platform
include ../make.def

EXES     = mm_testbed$(EXE) jac_solv_ocl_basic$(EXE) jac_solv_ocl_colmaj$(EXE) jac_solv_ocl_colmaj_nobr$(EXE) jac_solv_ocl_best$(EXE)

MM_OBJS  = mm_testbed.$(OBJ) mm_utils.$(OBJ) mm_ijk.$(OBJ) mm_tst_cases.$(OBJ)

JAC_OCL_BASIC_OBJS  = jac_solv_ocl_basic.$(OBJ) mm_utils.$(OBJ) ocl_utils.$(OBJ)

JAC_OCL_COLMAJ_OBJS  = jac_solv_ocl_colmaj.$(OBJ) mm_utils.$(OBJ) ocl_utils.$(OBJ)

JAC_OCL_COLMAJ_NOBR_OBJS  = jac_solv_ocl_colmaj_nobr.$(OBJ) mm_utils.$(OBJ) ocl_utils.$(OBJ)

JAC_OCL_BEST_OBJS  = jac_solv_ocl_best.$(OBJ) mm_utils.$(OBJ) ocl_utils.$(OBJ)


all: $(EXES)

jac_solv_ocl_basic$(EXE): $(JAC_OCL_BASIC_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o jac_solv_ocl_basic$(EXE) $(JAC_OCL_BASIC_OBJS) $(LIBS) $(OCL_LIBS)

jac_solv_ocl_colmaj$(EXE): $(JAC_OCL_COLMAJ_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o jac_solv_ocl_colmaj$(EXE) $(JAC_OCL_COLMAJ_OBJS) $(LIBS) $(OCL_LIBS)

jac_solv_ocl_colmaj_nobr$(EXE): $(JAC_OCL_COLMAJ_NOBR_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o jac_solv_ocl_colmaj_nobr$(EXE) $(JAC_OCL_COLMAJ_NOBR_OBJS) $(LIBS) $(OCL_LIBS)

jac_solv_ocl_best$(EXE): $(JAC_OCL_BEST_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o jac_solv_ocl_best$(EXE) $(JAC_OCL_BEST_OBJS) $(LIBS) $(OCL_LIBS)

mm_testbed$(EXE): $(MM_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o mm_testbed$(EXE) $(MM_OBJS) $(LIBS)

test: $(EXES)
	for i in $(EXES); do \
            $(PRE)$$i; \
        done

clean:
	$(RM) $(EXES) *.$(OBJ)

mm_testbed.$(OBJ): mm_utils.h
mm_utils.$(OBJ): mm_utils.h
mm_ijk.$(OBJ): mm_utils.h
mm_tst_cases.$(OBJ): mm_utils.h
jac_solv_ocl_basic.$(OBJ): mm_utils.h ocl_utils.h
jac_solv_ocl_colmaj.$(OBJ): mm_utils.h ocl_utils.h
jac_solv_ocl_colmaj_nobr.$(OBJ): mm_utils.h ocl_utils.h
jac_solv_ocl_best.$(OBJ): mm_utils.h ocl_utils.h
ocl_utils.$(OBJ): ocl_utils.h


.SUFFIXES:
.SUFFIXES: .c .cpp .$(OBJ)

.c.$(OBJ):
	$(CC) $(CFLAGS) -c $<

.cpp.$(OBJ):
	$(CC) $(CFLAGS) -c $<


