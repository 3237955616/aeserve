Start redis server with docker:
```bash
sudo docker run --name my-redis -p 6379:6379 -d redis
```

Install redis-py with `pip install redis`.

### Download AEServe and kvcached
```bash
# install AEServe
git clone https://github.com/LiRongchuan/AEServe.git

# install kvcached
git clone https://github.com/LiRongchuan/kvcached-research.git
```

```bash
sudo docker run -dit --gpus all --ipc=host --network=host \
    -v `pwd`/aeserve:/sgl-workspace/aeserve/ \
    -v `pwd`/kvcached-research:/sgl-workspace/kvcached \
    --name aeserve \
    lmsysorg/sglang:v0.3.4.post2-cu121 bash
```

```bash
cd /sgl-workspace/aeserve/python 
pip install -e .
cd /sgl-workspace/kvcached
pip install --no-build-isolation -e .
pip install tensordict --upgrade
pip install redis
pip install lm_eval
python3 setup.py clean
python3 setup.py install
```
```bash
cd /sgl-workspace/aeserve-public/benchmark/multi-model
python3 -m sglang.launch_multi_model_server --port 30000 --model-config-file ./model_configs/setup.json --disable-cuda-graph --disable-radix-cache --enable-controller --enable-cpu-share-memory --enable-elastic-memory --use-kvcached-v0 --policy resize-global --log-file ./server.log --async-loading

# Arena-trace测试，修改num-gpus和rate-scale
python3 benchmark.py \
  --base-url http://127.0.0.1:30000 \
  --real-trace ./real_trace.pkl \
  --model-ids 0 1 \
  --num-gpus 2 \
  --workload-scale 1 \
  --rate-scale 2 \
  --ttft-slo-scale 10 \
  --tpot-slo-scale 10

# SharedGPT测试，修改num-gpus和rate-scale

python3 benchmark.py \
  --base-url http://127.0.0.1:30000 \
  --sharedgpt ./sharedgpt/sharedgpt_n3_rate_1.json \
  --model-ids 0 1 \
  --num-gpus 4 \
  --ttft-slo-scale 10 \
  --tpot-slo-scale 10

# 消融测试，修改model_scheduler PRIORITY

```

