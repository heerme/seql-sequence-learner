CXX = g++
VERSION = 0.13
CXXFLAGS_DEBUG = -g -Wall -Wno-deprecated
CXXFLAGS = -O3 -Wall -Wno-deprecated
EXECPREFIX =
LDFLAGS  = 
TARGETS0 = seql_learn${EXEC_PREFIX}
TARGETS1 = seql_mkmodel${EXEC_PREFIX}
TARGETS2 = seql_classify${EXEC_PREFIX}
TARGETS3 = seql_classify_tune_threshold_min_errors${EXEC_PREFIX}

OBJ2 = str2node_string_symbol.o

all: seql_learn seql_mkmodel seql_classify seql_classify_tune_threshold_min_errors

seql_learn: seql_learn.o
	${CXX} ${CFLAGS} ${LDFLAGS} -o ${TARGETS0} seql_learn.o ${LDFLAGS}

seql_mkmodel: seql_mkmodel.o ${OBJ2}
	${CXX} ${CFLAGS} ${LDFLAGS} -o ${TARGETS1} seql_mkmodel.o ${LDFLAGS}

seql_classify: seql_classify.o ${OBJ2}
	${CXX} ${CFLAGS} ${LDFLAGS} -o ${TARGETS2} ${OBJ2} seql_classify.o ${LDFLAGS}

seql_classify_tune_threshold_min_errors: seql_classify_tune_threshold_min_errors.o ${OBJ2}
	${CXX} ${CFLAGS} ${LDFLAGS} -o ${TARGETS3} ${OBJ2} seql_classify_tune_threshold_min_errors.o ${LDFLAGS}

clean:
	rm -f *.o ${TARGETS0} ${TARGETS01} ${TARGETS1} ${TARGETS2} ${TARGETS3} core *~ *.tar.gz *.exe core* 

check:
test_word:	
	./seql_learn -n 0 -v 2 data/toy.word.train seql.toy.word.model
	echo
	./seql_mkmodel -i seql.toy.word.model -o seql.toy.word.model.bin -O seql.toy.word.model.predictors
	echo
	./seql_classify -n 0 -v 4 -t 0 data/toy.word.test seql.toy.word.model.bin
	
test_char:	
	./seql_learn -n 1 -v 2 data/toy.char.train seql.toy.char.model
	echo
	./seql_mkmodel -i seql.toy.char.model -o seql.toy.char.model.bin -O seql.toy.char.model.predictors
	echo
	./seql_classify -n 1 -v 4 -t 0 data/toy.char.test seql.toy.char.model.bin	

