#!/bin/bash
# Stage 3 rollback gates. Same launch shape as spec_bench.sh: rank 0 started LOCALLY on
# rank 0 and the workers reached over the private fabric (10.10.10.4/.6), never through the
# public brev gateway — the gateway-tunnelled variant intermittently loses the coord
# bootstrap ("accept timeout" / "connect: Connection refused").
set -u
ROOT=/home/victor/work/k3-tp3-0012; DIST=$ROOT/dist; RUN=$ROOT/run
MODEL=/home/victor/models/kimi-k3-neuron-iq1s-local/k3-neuron-iq1s-00001-of-00009.gguf
R1=10.10.10.4; R2=10.10.10.6
SSHQ="ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no"
K=${K:-8}
CONT=${CONT:-3}
PROMPT=${PROMPT:-1,2,3}
TAG=${TAG:-default}
EXTRA=${EXTRA:-}
mkdir -p "$RUN"
pkill -9 -f "[k]imi_k3_dist_generate" 2>/dev/null
$SSHQ victor@$R1 "pkill -9 -f '[k]imi_k3_dist_generate'" 2>/dev/null
$SSHQ victor@$R2 "pkill -9 -f '[k]imi_k3_dist_generate'" 2>/dev/null
sleep 2
# PREFILL_CHUNK bounds the batched scratch, so it must be >= K or forward_batch refuses.
COMMON="LD_LIBRARY_PATH=$DIST SPARKINFER_K3_MOE_WEPS=0 SPARKINFER_K3_GRAPH=0 NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0 SPARKINFER_K3_PREFILL_CHUNK=16 $EXTRA"
ARGS="--model $MODEL --prompt-ids $PROMPT --max-ctx 4096 --spec-rollback $K --spec-cont $CONT"
( cd $ROOT && env $COMMON nohup $DIST/kimi_k3_dist_generate --rank 0 --world 3 --listen 0.0.0.0:29500 \
    $ARGS > $RUN/roll_${TAG}_r0.stdout 2> $RUN/roll_${TAG}_r0.stderr < /dev/null & echo $! > $RUN/roll_r0.pid )
sleep 6
$SSHQ victor@$R1 "cd $ROOT && env $COMMON nohup $DIST/kimi_k3_dist_generate --rank 1 --world 3 --coord 10.10.10.2:29500 $ARGS > $RUN/roll_${TAG}_r1.stdout 2> $RUN/roll_${TAG}_r1.stderr < /dev/null & disown" 2>/dev/null &
$SSHQ victor@$R2 "cd $ROOT && env $COMMON nohup $DIST/kimi_k3_dist_generate --rank 2 --world 3 --coord 10.10.10.2:29500 $ARGS > $RUN/roll_${TAG}_r2.stdout 2> $RUN/roll_${TAG}_r2.stderr < /dev/null & disown" 2>/dev/null &
echo "launched: K=$K CONT=$CONT TAG=$TAG EXTRA='$EXTRA'"
