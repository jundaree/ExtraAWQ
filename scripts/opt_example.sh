MODELS=( "opt-1.3b" "opt-2.7b" "opt-6.7b" "opt-13b" )
for MODEL in "${MODELS[@]}"; do
    echo "Processing model: $MODEL"
    
    # run AWQ search
    python -m awq.entry --model_path ../dataset/$MODEL \
        --w_bit 4 --q_group_size 128 \
        --run_awq --dump_awq ../awq_cache/$MODEL-w4-g128.pt
    
    
    echo "Completed processing for $MODEL"
    echo "----------------------------------------"
done

for MODEL in "${MODELS[@]}"; do

    # evaluate the AWQ quantize model (simulated pseudo quantization)
    python -m awq.entry --model_path ../dataset/$MODEL \
        --tasks wikitext \
        --w_bit 4 --q_group_size 128 \
        --load_awq ../awq_cache/$MODEL-w4-g128.pt \
        --q_backend fake

    # generate real quantized weights (w4)
    python -m awq.entry --model_path ../dataset/$MODEL \
        --w_bit 4 --q_group_size 128 \
        --load_awq ../awq_cache/$MODEL-w4-g128.pt \
        --q_backend real --dump_quant ../quant_cache/$MODEL-w4-g128-awq.pt

    # load and evaluate the real quantized model (smaller gpu memory usage)
    python -m awq.entry --model_path ../dataset/$MODEL \
        --tasks wikitext \
        --w_bit 4 --q_group_size 128 \
        --load_quant ../quant_cache/$MODEL-w4-g128-awq.pt

done