archive_path=$1

test_data=$2

generated_dirpath=$3

prefix="train"

beam_size=4

batch_size=128

for ckpt in {1..5};
do
  mkdir $archive_path"/output_e${ckpt}/"
  allennlp eval $archive_path $test_data \
          --include-package allen_modules \
          --weights-file $archive_path"/model_state_e${ckpt}_b0.th" \
          --output-file $archive_path"/output_e${ckpt}/out.${prefix}.metrics" \
          --predictions-output-file $archive_path"/output_e${ckpt}/out.${prefix}.pred" \
          --cuda-device 0 \
          --batch-size $batch_size \
          --overrides '{"model.beam_search.beam_size": '$beam_size', "validation_data_loader":{"batch_sampler": {"type": "bucket", "sorting_keys": ["target_tokens"] } } }'
done

python -m utils.discrim.generate_negative_samples \
        $archive_path \
        $generated_dirpath \
        --negative_ratio 3
