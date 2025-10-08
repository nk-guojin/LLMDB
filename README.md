# LLMDB

Code for the Paper "Zero-Shot Large Language Model Recommender on Mitigating"

## Quick Start

1. Download processed atomic .inter and .item files from recbole.
    https://github.com/RUCAIBox/RecSysDatasets

2. Evaluate LLMDB on ML-100K dataset.
    ```bash
    cd LLMDB
    python main.py -m LLMDB -d ml-100k -u 200
    ```
