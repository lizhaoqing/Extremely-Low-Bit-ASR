#!/usr/bin/env bash

# QACT: Quantization-Aware Co-Training for Conformer ASR
# This script should be placed inside the ESPnet Switchboard recipe directory.
# See README.md for setup instructions.

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

export CUDA_VISIBLE_DEVICES=0

# general configuration
backend=pytorch
stage=4        # start from stage 4 (training) if data is already prepared
stop_stage=5
ngpu=1
debugmode=1
dumpdir=dump
N=0
verbose=0
n_iter_processes=64
resume=

# feature configuration
do_delta=false

# QACT configuration
preprocess_config=conf/specaug.yaml
train_config=conf/train_qact.yaml

# decode configuration
# Change this to decode at different precisions:
#   decode_1bit.yaml     : all 1-bit
#   decode_2bit.yaml     : all 2-bit
#   decode_1.5bit.yaml   : 1.5-bit mixed precision
decode_config=conf/decode_2bit.yaml

# rnnlm related
lm_resume=
lmtag=

# decoding parameter
n_average=10
use_valbest_average=true

# bpemode
nbpe=2000
bpemode=bpe

# exp tag
tag="" # tag for managing experiments

. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

train_set=train_nodup_trim
train_dev=train_dev_trim
recog_set="eval2000 rt02 rt03"

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

dict=data/lang_char/train_nodup_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/train_nodup_${bpemode}${nbpe}

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

# ========================== Stage 4: Training ==========================
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: QACT Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --n-iter-processes ${n_iter_processes} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi

# ========================== Stage 5: Decoding ==========================
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding with config: ${decode_config}"
    nj=32

    # Average ASR models
    if ${use_valbest_average}; then
        recog_model=model.val${n_average}.avg.best
        opt="--log ${expdir}/results/log"
    else
        recog_model=model.last${n_average}.avg.best
        opt="--log"
    fi

    python average.py --backend ${backend} \
                    --snapshots ${expdir}/results/snapshot.ep.* \
                    --out ${expdir}/results/${recog_model} \
                    --log ${opt} \
                    --num ${n_average}

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json
        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            python recog/asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}

        # scoring
        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
        if [[ "${decode_dir}" =~ "eval2000" ]]; then
            local/score_sclite.sh data/eval2000 ${expdir}/${decode_dir}
        elif [[ "${decode_dir}" =~ "rt03" ]]; then
            local/score_sclite.sh data/rt03 ${expdir}/${decode_dir}
        elif [[ "${decode_dir}" =~ "rt02" ]]; then
            local/score_sclite.rt02.sh data/rt02 ${expdir}/${decode_dir}
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed."
    echo "Finished"
fi
