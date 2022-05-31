	
#!/bin/bash

input_dir=$1
output_dir=$2



python main_get_vad.py --wav_dir $input_dir --output_dir $output_dir --mode 0 --hoplength 30 || exit 1




exit 0
