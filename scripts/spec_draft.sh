#!/bin/bash
# Stage 4 end-to-end speculative decode. Same launch shape as spec_rollback.sh: rank 0
# started LOCALLY on rank 0 and the workers reached over the private fabric
# (10.10.10.4/.6), never through the public brev gateway — the gateway-tunnelled variant
# intermittently loses the coord bootstrap ("accept timeout" / "connect: Connection
# refused").
#
# Unlike --spec-bench/--spec-rollback this DOES use the control plane: rank 0 chooses
# drafts from its own generation history and broadcasts them. Workers still need K on the
# command line so they arm the rollback ring and accept K-token steps.
#
#   K=0            baseline (no drafting), the comparison arm
#   K=2|4|8        speculative, batch width K
set -u
ROOT=/home/victor/work/k3-tp3-0012; DIST=$ROOT/dist; RUN=$ROOT/run
MODEL=/home/victor/models/Kimi-K3-UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00009.gguf
R1=10.10.10.4; R2=10.10.10.6
SSHQ="ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no"
K=${K:-0}
NPRED=${NPRED:-64}
TAG=${TAG:-base}
PF=${PF:-$ROOT/p4/prompts.txt}
NMIN=${NMIN:-1}
NMAX=${NMAX:-3}
EXTRA=${EXTRA:-}
mkdir -p "$RUN"
pkill -9 -f "[k]imi_k3_dist_generate" 2>/dev/null
$SSHQ victor@$R1 "pkill -9 -f '[k]imi_k3_dist_generate'" 2>/dev/null
$SSHQ victor@$R2 "pkill -9 -f '[k]imi_k3_dist_generate'" 2>/dev/null
sleep 2
# PREFILL_CHUNK bounds the batched scratch, so it must be >= K or forward_batch refuses.
COMMON="LD_LIBRARY_PATH=$DIST SPARKINFER_K3_MOE_WEPS=0 SPARKINFER_K3_GRAPH=0 NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0 SPARKINFER_K3_PREFILL_CHUNK=16 $EXTRA"
SPEC=""
if [ "$K" -gt 0 ]; then SPEC="--spec-draft $K --spec-ngram-min $NMIN --spec-ngram-max $NMAX"; fi
ARGS="--model $MODEL --prompts-file $PF --max-ctx 4096 --n-predict $NPRED $SPEC"
( cd $ROOT && env $COMMON nohup $DIST/kimi_k3_dist_generate --rank 0 --world 3 --listen 0.0.0.0:29500 \
    $ARGS > $RUN/draft_${TAG}_r0.stdout 2> $RUN/draft_${TAG}_r0.stderr < /dev/null & echo $! > $RUN/draft_r0.pid )
sleep 6
$SSHQ victor@$R1 "cd $ROOT && env $COMMON nohup $DIST/kimi_k3_dist_generate --rank 1 --world 3 --coord 10.10.10.2:29500 $ARGS > $RUN/draft_${TAG}_r1.stdout 2> $RUN/draft_${TAG}_r1.stderr < /dev/null & disown" 2>/dev/null &
$SSHQ victor@$R2 "cd $ROOT && env $COMMON nohup $DIST/kimi_k3_dist_generate --rank 2 --world 3 --coord 10.10.10.2:29500 $ARGS > $RUN/draft_${TAG}_r2.stdout 2> $RUN/draft_${TAG}_r2.stderr < /dev/null & disown" 2>/dev/null &
echo "launched: K=$K TAG=$TAG NPRED=$NPRED NMIN=$NMIN NMAX=$NMAX"
