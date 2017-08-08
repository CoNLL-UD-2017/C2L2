# CoNLL'17 - C2L2

This is the repo for the code to reproduce the experiments and results of the system entry C2L2 in the CoNLL shared task.

For more up-to-date parser under maintenance, please see <https://github.com/tzshi/cdparser>.

# Requirements

- [DyNet](https://github.com/clab/dynet) 1.1 (tested with commit b7e4f4e1d9129b6b4cef8c3c85da58dac5adc392 )

# Documentation

Example training script calling from bash:

```
MKL_NUM_THREADS=2 python -m cdparser_multi.cdparser \
- build-vocab $TRAIN_FILE \
- create-parser \
- init-model \
- train $TRAIN_FILE --dev $DEV_FILE \
    --utag True --mst True --aedp True --ahdp True \
- finish --dynet-mem 2000
```

Example test script calling from Python environment:

```
parser = CDParser()
parser.load_model(model_file, verbose=False)
parser.predict(graphs, aedp=True)
```

# Citation

If you make use of this software in your research, we appreciate you citing the following papers:

```
@InProceedings{shi-etal2017conll,
    author    = {Shi, Tianze  and  Wu, Felix G.  and  Chen, Xilun  and  Cheng, Yao},
    title     = {Combining Global Models for Parsing Universal Dependencies},
    booktitle = {Proceedings of the CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies},
    month     = {August},
    year      = {2017},
    address   = {Vancouver, Canada},
    publisher = {Association for Computational Linguistics},
    pages     = {31--39},
    url       = {http://www.aclweb.org/anthology/K17-3003}
}
```

```
@InProceedings{shi+huang+lee2017exact,
    author    = {Shi, Tianze  and  Huang, Liang  and  Lee, Lillian},
    title     = {Fast(er) Exact Decoding and Global Training for Transition-based Dependency Parsing via a Minimal Feature Set},
    booktitle = {Proceedings of the Conference on Empirical Methods in Natural Language Processing},
    month     = {September},
    year      = {2017},
    address   = {Copenhagen, Denmark},
    publisher = {Association for Computational Linguistics},
    pages     = {(To appear)}
}
```

# Acknowledgement

When implementing the first-order graph-based algorithm, we referenced the BiST parser: <https://github.com/elikip/bist-parser>.
