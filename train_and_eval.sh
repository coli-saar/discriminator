config=$1

archive_name="$(basename -s .jsonnet $config)"

config_dir="$(dirname $config)"

sub_config_dir="${config_dir#configs/}/${archive_name}"

test_data=$2

seed=$3

mkdir -p "model_archives/${sub_config_dir}/"

echo "Training model with seed ${seed} Save to model_archives/${sub_config_dir}/${seed}"

archive_path="model_archives/${sub_config_dir}/${seed}"

overrides='{random_seed:'$seed',numpy_seed:'$seed',pytorch_seed:'$seed'}'

TOKENIZERS_PARALLELISM=false allennlp train $config -s $archive_path \
          -f \
          --include-package allen_modules \
          --file-friendly \
          -o $overrides

TOKENIZERS_PARALLELISM=false allennlp eval $archive_path $test_data \
          --include-package allen_modules \
          --output-file $archive_path"/output/out.test.metrics" \
          --predictions-output-file $archive_path"/output/out.test.pred" \
          --cuda-device 0 \
          --batch-size 64 \
          --overrides '{"model.beam_search.beam_size": 4}'
 
#allennlp eval $archive_path $test_data \
#          --include-package allen_modules \
#          --output-file $archive_path"/output/out.test.metrics" \
#          --predictions-output-file $archive_path"/output/out.test.pred" \
#          --cuda-device 0 \
#          --batch-size 32 \
#          --overrides '{"model.beam_search.beam_size": 4}'