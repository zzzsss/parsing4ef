#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import logging
import os.path
import sys
import multiprocessing
 
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    inp, outp = sys.argv[1:3]

    # special one 
    model = Word2Vec(LineSentence(inp), size=50, window=7, min_count=10, workers=2,
                     sg=1, sample=1e-5, negative=15)    # skip-gram + negative sample

    # trim unneeded model memory = use(much) less RAM
    #model.init_sims(replace=True)
    model.save_word2vec_format(outp, binary=False)
