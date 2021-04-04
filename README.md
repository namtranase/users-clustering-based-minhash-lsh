# users-clustering-based-minhash-lsh

Implement user clusters based on `MinHash LSH algorithm`.
User clusters could server for many kind of application, such
as Recommendation System, Advertising System ...

Code is implement by the idea in this paper below, please see
the paper to understand the algorithm:
[PDF] (https://www2007.org/papers/paper570.pdf)

## Create virtual environment and install requirement libs.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run validate user clusters program
```bash
PYTHONPATH=. ./bin/validate_user_clusters
```
