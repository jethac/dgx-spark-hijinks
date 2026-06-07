# AEON Qwen3.6 DFlash Reproduction Attempt

Timestamp: 2026-06-08T05:01 JST

Status: blocked before serving; model and drafter downloaded, but Docker image pull did not register within bounded retries.

## Model Cache

22G	/home/jethac/models/aeon/qwen36-nvfp4
905M	/home/jethac/models/aeon/qwen36-dflash

## Image State

image_not_registered
[]
Error response from daemon: No such image: ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2

## Process State

## Docker Images
ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2 sha256:0b93829450c7b2afcd6a81e8cb9815b1ea62ea8eb2a00bdc0badebcc9f725c88 5f55c2af4593 24GB

## Last Pull Log Lines
8d332f2ae9ee: Waiting
42839964e8f7: Pulling fs layer
8717b1e6b67f: Pulling fs layer
a29aaa8aa7ac: Waiting
41126d95418d: Waiting
75ae7f50d885: Waiting
27942cf0a22d: Waiting
42839964e8f7: Waiting
91529bbe5dbc: Waiting
8717b1e6b67f: Waiting
719ea8d898ae: Waiting
1bfe1a4ad78a: Waiting
ff448999479a: Waiting
e7da47737d89: Waiting
fd893f316d5f: Waiting
1e88f0ae45ae: Waiting
18218423db25: Verifying Checksum
18218423db25: Download complete
2b9585699ff1: Verifying Checksum
2b9585699ff1: Download complete
3fa23874706e: Download complete
314a2f652b6c: Verifying Checksum
314a2f652b6c: Download complete
2b9585699ff1: Pull complete
18218423db25: Pull complete
09e9408034f3: Verifying Checksum
09e9408034f3: Download complete
e5e5f1aa8be2: Verifying Checksum
e5e5f1aa8be2: Download complete
949f935e0756: Verifying Checksum
949f935e0756: Download complete
6f596c6704f1: Verifying Checksum
6f596c6704f1: Download complete
067f10bdf22e: Verifying Checksum
067f10bdf22e: Download complete
4f4fb700ef54: Download complete
9e4aabb282ff: Verifying Checksum
9e4aabb282ff: Download complete
067f10bdf22e: Pull complete
09e9408034f3: Pull complete
3fa23874706e: Pull complete
314a2f652b6c: Pull complete
c744a032a83b: Verifying Checksum
c744a032a83b: Download complete
5f97a61844b9: Verifying Checksum
5f97a61844b9: Download complete
d8413400c3ea: Verifying Checksum
d8413400c3ea: Download complete
15039e7d116d: Verifying Checksum
15039e7d116d: Download complete
15039e7d116d: Pull complete
e5e5f1aa8be2: Pull complete
949f935e0756: Pull complete
6f596c6704f1: Pull complete
32cc32da7083: Verifying Checksum
32cc32da7083: Download complete
0759b5cda1f1: Verifying Checksum
0759b5cda1f1: Download complete
05ca27917f0f: Verifying Checksum
05ca27917f0f: Download complete
ccf3ccc257e5: Verifying Checksum
ccf3ccc257e5: Download complete
cfc2d9b90a4a: Verifying Checksum
cfc2d9b90a4a: Download complete
fab53726b0d0: Verifying Checksum
fab53726b0d0: Download complete
fbba2abb37b9: Verifying Checksum
fbba2abb37b9: Download complete
a74b4298415c: Verifying Checksum
a74b4298415c: Download complete
210178c15afc: Verifying Checksum
210178c15afc: Download complete
4a4a09129125: Verifying Checksum
4a4a09129125: Download complete
32cc32da7083: Pull complete
4f4fb700ef54: Pull complete
9e4aabb282ff: Pull complete
a1ec79daa4f2: Verifying Checksum
a1ec79daa4f2: Download complete
c744a032a83b: Pull complete
5f97a61844b9: Pull complete
d8413400c3ea: Pull complete
a29aaa8aa7ac: Verifying Checksum
a29aaa8aa7ac: Download complete
27942cf0a22d: Verifying Checksum
27942cf0a22d: Download complete
91529bbe5dbc: Verifying Checksum
91529bbe5dbc: Download complete
1bfe1a4ad78a: Verifying Checksum
1bfe1a4ad78a: Download complete
e7da47737d89: Verifying Checksum
e7da47737d89: Download complete
719ea8d898ae: Verifying Checksum
719ea8d898ae: Download complete
41126d95418d: Verifying Checksum
41126d95418d: Download complete
75ae7f50d885: Verifying Checksum
75ae7f50d885: Download complete
ff448999479a: Verifying Checksum
ff448999479a: Download complete
1e88f0ae45ae: Download complete
fd893f316d5f: Verifying Checksum
fd893f316d5f: Download complete
42839964e8f7: Verifying Checksum
42839964e8f7: Download complete
8717b1e6b67f: Verifying Checksum
8717b1e6b67f: Download complete
9be51ddd2646: Verifying Checksum
9be51ddd2646: Download complete
9be51ddd2646: Pull complete
a1ec79daa4f2: Pull complete
0759b5cda1f1: Pull complete
05ca27917f0f: Pull complete
ccf3ccc257e5: Pull complete
cfc2d9b90a4a: Pull complete
fab53726b0d0: Pull complete
fbba2abb37b9: Pull complete
a74b4298415c: Pull complete
210178c15afc: Pull complete
4a4a09129125: Pull complete
