#
#  USAGE:
#     make          ... to build the program
#     make test     ... to run the default test case
#

# verify that you are including the right make.def file for the platform
include make.def

EXES     = mm_testbed$(EXE) jac_solv$(EXE) jac_solv_ocl$(EXE) vadd$(EXE) deviceinfo$(EXE)

MM_OBJS  = mm_testbed.$(OBJ) mm_utils.$(OBJ) mm_ijk.$(OBJ) mm_tst_cases.$(OBJ)


JAC_OBJS  = jac_solv.$(OBJ) mm_utils.$(OBJ)

JAC_OCL_OBJS  = jac_solv_ocl.$(OBJ) mm_utils.$(OBJ) ocl_utils.$(OBJ)

VADD_OCL_OBJS  = vadd.$(OBJ) ocl_utils.$(OBJ)

DEVICE_OBJS	= deviceinfo.$(OBJ) ocl_utils.$(OBJ)

all: $(EXES)

jac_solv$(EXE): $(JAC_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o jac_solv$(EXE) $(JAC_OBJS) $(LIBS)

mm_testbed$(EXE): $(MM_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o mm_testbed$(EXE) $(MM_OBJS) $(LIBS)

vadd$(EXE): $(VADD_OCL_OBJS)
	$(CLINKER) $(CFLAGS) -o vadd$(EXE) $(VADD_OCL_OBJS) $(LIBS) $(OCL_LIBS)

jac_solv_ocl$(EXE): $(JAC_OCL_OBJS)
	$(CLINKER) $(CFLAGS) -o jac_solv_ocl$(EXE) $(JAC_OCL_OBJS) $(LIBS) $(OCL_LIBS)

deviceinfo$(EXE): $(DEVICE_OBJS)
	$(CLINKER) $(CFLAGS) -o deviceinfo$(EXE) $(DEVICE_OBJS) $(LIBS) $(OCL_LIBS)

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
jac_solv.$(OBJ): mm_utils.h
jac_solv_ocl.$(OBJ): mm_utils.h ocl_utils.h
vadd.$(OBJ): ocl_utils.h
ocl_utils.$(OBJ): ocl_utils.h


.SUFFIXES:
.SUFFIXES: .c .cpp .$(OBJ)

.c.$(OBJ):
	$(CC) $(CFLAGS) -c $<

.cpp.$(OBJ):
	$(CC) $(CFLAGS) -c $<


