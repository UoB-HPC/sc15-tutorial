#
#  USAGE:
#     make          ... to build the program
#     make test     ... to run the default test case
#

# verify that you are including the right make.def file for the platform
include make.def

EXES     = mm_testbed$(EXE) cmass$(EXE) gauss_solv$(EXE) stencil$(EXE) jac_solv$(EXE) jac_solv_ocl_basic$(EXE) jac_solv_ocl_colmaj$(EXE) jac_solv_ocl_colmaj_nobr$(EXE) jac_solv_ocl_colmaj_nobr_wg$(EXE) jac_solv_ocl_best$(EXE) vadd$(EXE)

MM_OBJS  = mm_testbed.$(OBJ) mm_utils.$(OBJ) mm_ijk.$(OBJ) mm_tst_cases.$(OBJ)

CM_OBJS  = center_of_mass.$(OBJ) random.$(OBJ)

STN_OBJS  = stencil.$(OBJ)

GAUS_OBJS  = gauss_solv.$(OBJ) mm_utils.$(OBJ)

JAC_OBJS  = jac_solv.$(OBJ) mm_utils.$(OBJ)

JAC_OCL_BASIC_OBJS  = jac_solv_ocl_basic.$(OBJ) mm_utils.$(OBJ) ocl_utils.$(OBJ)

JAC_OCL_COLMAJ_OBJS  = jac_solv_ocl_colmaj.$(OBJ) mm_utils.$(OBJ) ocl_utils.$(OBJ)

JAC_OCL_COLMAJ_NOBR_OBJS  = jac_solv_ocl_colmaj_nobr.$(OBJ) mm_utils.$(OBJ) ocl_utils.$(OBJ)

JAC_OCL_COLMAJ_NOBR_WG_OBJS  = jac_solv_ocl_colmaj_nobr_wg.$(OBJ) mm_utils.$(OBJ) ocl_utils.$(OBJ)

JAC_OCL_BEST_OBJS  = jac_solv_ocl_best.$(OBJ) mm_utils.$(OBJ) ocl_utils.$(OBJ)

VADD_OCL_OBJS  = vadd.$(OBJ) ocl_utils.$(OBJ)

all: $(EXES)

stencil$(EXE): $(STN_OBJS)
	$(CLINKER) $(CFLAGS) -o stencil$(EXE) $(STN_OBJS) $(LIBS)

jac_solv$(EXE): $(JAC_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o jac_solv$(EXE) $(JAC_OBJS) $(LIBS)

jac_solv_ocl_basic$(EXE): $(JAC_OCL_BASIC_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o jac_solv_ocl_basic$(EXE) $(JAC_OCL_BASIC_OBJS) $(LIBS) $(OCL_LIBS)

jac_solv_ocl_colmaj$(EXE): $(JAC_OCL_COLMAJ_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o jac_solv_ocl_colmaj$(EXE) $(JAC_OCL_COLMAJ_OBJS) $(LIBS) $(OCL_LIBS)

jac_solv_ocl_colmaj_nobr$(EXE): $(JAC_OCL_COLMAJ_NOBR_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o jac_solv_ocl_colmaj_nobr$(EXE) $(JAC_OCL_COLMAJ_NOBR_OBJS) $(LIBS) $(OCL_LIBS)

jac_solv_ocl_colmaj_nobr_wg$(EXE): $(JAC_OCL_COLMAJ_NOBR_WG_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o jac_solv_ocl_colmaj_nobr_wg$(EXE) $(JAC_OCL_COLMAJ_NOBR_WG_OBJS) $(LIBS) $(OCL_LIBS)

jac_solv_ocl_best$(EXE): $(JAC_OCL_BEST_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o jac_solv_ocl_best$(EXE) $(JAC_OCL_BEST_OBJS) $(LIBS) $(OCL_LIBS)

gauss_solv$(EXE): $(GAUS_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o gauss_solv$(EXE) $(GAUS_OBJS) $(LIBS)

mm_testbed$(EXE): $(MM_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o mm_testbed$(EXE) $(MM_OBJS) $(LIBS)

cmass$(EXE): $(CM_OBJS) random.h
	$(CLINKER) $(CFLAGS) -o cmass$(EXE) $(CM_OBJS) $(LIBS)

vadd$(EXE): $(VADD_OCL_OBJS)
	$(CLINKER) $(CFLAGS) -o vadd$(EXE) $(VADD_OCL_OBJS) $(LIBS) $(OCL_LIBS)

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
gauss_solv.$(OBJ): mm_utils.h
jac_solv.$(OBJ): mm_utils.h
jac_solv_ocl_basic.$(OBJ): mm_utils.h ocl_utils.h
jac_solv_ocl_colmaj.$(OBJ): mm_utils.h ocl_utils.h
jac_solv_ocl_colmaj_nobr.$(OBJ): mm_utils.h ocl_utils.h
jac_solv_ocl_colmaj_nobr_wg.$(OBJ): mm_utils.h ocl_utils.h
jac_solv_ocl_best.$(OBJ): mm_utils.h ocl_utils.h
vadd.$(OBJ): ocl_utils.h
ocl_utils.$(OBJ): ocl_utils.h


.SUFFIXES:
.SUFFIXES: .c .cpp .$(OBJ)

.c.$(OBJ):
	$(CC) $(CFLAGS) -c $<

.cpp.$(OBJ):
	$(CC) $(CFLAGS) -c $<


