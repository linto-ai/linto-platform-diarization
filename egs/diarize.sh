#!/bin/bash



start=$(date +%s.%N)

currDir=$PWD
kaldiDir=/home/wghezaiel/kaldi
expDir=$currDir/exp
wavDir=$currDir/in_wavs
labDir=$currDir/lab
rttmDir=$currDir/rttm
wavList=$currDir/wavList
readlink -f $wavDir/* > $wavList

# Extraction parameters
window=1.5
window_period=0.75
min_segment=0.5
modelDir=$currDir/model_xvec



# Evaluation parameters
modelType=xvecTDNN
layerName=fc3
method=SC
useCollar=0
skipDataPrep=0
dataDir=$expDir/data
nj=1
n_spk=$1
max_spk=$2

bash run_vad.sh $wavDir $labDir

for label in $(ls $labDir); do
   
   cd $labDir
   
   base=$(basename $label .sad)
   

   
   rttm_out=$rttmDir/${base}.rttm
   grep 'sp' ${base}.sad | awk -v fname=$base '{print "SPEAKER" " " fname " " 1  " " $1  " " $2-$1 " " "<NA>" " " "<NA>"  " " $3  " "    "<NA>"}'   > $rttm_out
   rm ${base}.sad
done   

 
cd .. 

  


if [[ "$useCollar" == "1" ]]; then
  collarCmd="--collar 0.25 --ignore_overlaps"
else
  collarCmd="--ignore_overlaps"
fi

. ./cmd.sh
. ./path.sh

# Kaldi directory preparation
if [ "$skipDataPrep" == "0" ]; then

  rm -rf $expDir; mkdir -p $dataDir
  paste -d ' ' <(rev $wavList | cut -f 1 -d '/' | rev | sed "s/\.wav$/-rec/g") \
    <(cat $wavList | xargs readlink -f) > $dataDir/wav.scp
  paste -d ' ' <(cut -f 1 -d ' ' $dataDir/wav.scp | sed "s/-rec$//g") \
    <(cut -f 1 -d ' ' $dataDir/wav.scp | sed "s/-rec$//g") > $dataDir/utt2spk
  cp $dataDir/utt2spk $dataDir/spk2utt
  numUtts=`wc -l $dataDir/utt2spk | cut -f 1 -d ' '`
  paste -d ' ' <(cut -f 1 -d ' ' $dataDir/utt2spk) \
    <(cut -f 1 -d ' ' $dataDir/wav.scp) <(yes "0" | head -n $numUtts) <(cat $wavList | xargs soxi -D) \
    >  $dataDir/segments
  for rttmFile in $rttmDir/*.rttm; do
    #n=`cut -f 8 -d ' ' $rttmFile | sort | uniq | wc -l`
    #n = 4
    echo "$n_spk"
    echo "`basename $rttmFile .rttm` $n_spk" >> $dataDir/reco2num_spk
  done


  # Create VAD directory
  echo "Creating VAD files.."
  python convert_rttm_to_vad.py $wavDir $rttmDir $expDir/oracleVAD
  while read -r line; do
      uttID=`echo $line | cut -f 1 -d ' '`
      inVadFile=$expDir/oracleVAD/$uttID.csv
      [ ! -f $inVadFile ] && { echo "Input vad file does not exist"; exit 0; }
      paste -d ' ' <(echo $uttID) <(cut -f 2 -d ',' $inVadFile | tr "\n" " " | sed "s/^/ [ /g" | sed "s/$/ ]/g") >> $dataDir/vad.txt
  done < $dataDir/utt2spk
  copy-vector ark,t:$dataDir/vad.txt ark,scp:$dataDir/vad.ark,$dataDir/vad.scp
  echo "Done"

  # Feature preparation pipeline, until train_combined_no_sil
  utils/fix_data_dir.sh $dataDir
  steps/make_mfcc.sh --nj $nj \
                 --cmd "$train_cmd" \
                 --mfcc-config conf/mfcc.conf \
                 --write-utt2num-frames true \
                 $dataDir || exit 1
  utils/fix_data_dir.sh $dataDir

  diarization/vad_to_segments.sh --nj $nj \
                 --cmd "$train_cmd" \
                 --segmentation-opts '--silence-proportion 0.01001' \
                 --min-duration 0.5 \
                 $dataDir $dataDir/segmented || exit 1

  local/nnet3/xvector/prepare_feats.sh --nj $nj \
                 --cmd "$train_cmd" \
                 $dataDir/segmented \
                 $dataDir/segmented_cmn \
                 $dataDir/segmented_cmn/feats || exit 1
  cp $dataDir/segmented/segments $dataDir/segmented_cmn/segments
  utils/fix_data_dir.sh $dataDir/segmented_cmn
  utils/split_data.sh $dataDir/segmented_cmn $nj

  # Compute the subsegments directory
  utils/data/get_uniform_subsegments.py \
               --max-segment-duration=$window \
               --overlap-duration=$(perl -e "print ($window-$window_period);") \
               --max-remaining-duration=$min_segment \
               --constant-duration=True \
              $dataDir/segmented_cmn/segments > $dataDir/segmented_cmn/subsegments
  utils/data/subsegment_data_dir.sh $dataDir/segmented_cmn \
    $dataDir/segmented_cmn/subsegments $dataDir/pytorch_xvectors/subsegments
  utils/split_data.sh $dataDir/pytorch_xvectors/subsegments $nj
  
  
  diarization/nnet3/xvector/extract_xvectors.sh --nj $nj \
               --apply-cmn false \
               --min-segment $min_segment \
               --chunk-size 500 \
               $modelDir \
               $dataDir/pytorch_xvectors/subsegments \
               $dataDir/kaldi_xvectors/
  

  #for f in segments utt2spk spk2utt; do
   # cp $dataDir/pytorch_xvectors/subsegments/$f $dataDir/pytorch_xvectors/$f
  #done

else
  [ ! -f $dataDir/pytorch_xvectors/xvector.scp ] && echo "Cannot find features" && exit 1;
fi
utils/combine_data.sh $dataDir/kaldi_xvectors/   

if [ "$method" == "plda" ]; then

  diarization/nnet3/xvector/score_plda.sh --nj 1 \
               --cmd "$train_cmd" \
               $modelDir \
               $dataDir/kaldi_xvectors \
               $expDir/plda/scoring

  

  diarization/cluster.sh --nj 1 \
               --cmd "$train_cmd" \
               --threshold 0 \
               $expDir/plda/scoring \
               $expDir/plda/clustering_estNumSpkr

else

  # Compute the cosine affinity
  cd Auto-Tuning-Spectral-Clustering/sc_utils
  rm -rf $expDir/SC/cos_scores
  bash score_embedding.sh --cmd "$train_cmd" --nj 1 \
               --score_metric cos --out_dir $expDir/SC/cos_scores  \
               $dataDir/kaldi_xvectors $expDir/SC/cos_scores || exit 1
  cd ..

  rm -rf $expDir/SC/labels_estNumSpkr
  if [ -z "$max_spk" ]; then
	  if [ -z "$n_spk" ]; then
		  python spectral_opt.py --affinity_score_file $expDir/SC/cos_scores/scores.scp \
			       --threshold 0.05 --score_metric "cos" --max_speaker 10 \
			       --spt_est_thres 'None' \
			       --segment_file_input_path $dataDir/kaldi_xvectors/segments \
			       --spk_labels_out_path $expDir/SC/labels_estNumSpkr \
			       --sparse_search True || exit 1
		  mkdir -p $expDir/SC/clustering_estNumSpkr
		  
	  
	  else
		  python sc_utils/make_rttm.py $dataDir/kaldi_xvectors/segments \
		    $expDir/SC/labels_estNumSpkr $expDir/SC/clustering_estNumSpkr/rttm
		  python spectral_opt.py --affinity_score_file $expDir/SC/cos_scores/scores.scp \
			       --threshold 0.05 --score_metric "cos" --max_speaker 10 \
			       --spt_est_thres 'None' --reco2num_spk $dataDir/reco2num_spk \
			       --segment_file_input_path $dataDir/kaldi_xvectors/segments \
			       --spk_labels_out_path $expDir/SC/labels_estNumSpkr \
			       --sparse_search True || exit 1
		  mkdir -p $expDir/SC/clustering_estNumSpkr
		  python sc_utils/make_rttm.py $dataDir/kaldi_xvectors/segments \
		    $expDir/SC/labels_estNumSpkr $expDir/SC/clustering_estNumSpkr/rttm
	  fi
  else
	  if [ -z "$n_spk" ]; then
		  python spectral_opt.py --affinity_score_file $expDir/SC/cos_scores/scores.scp \
			       --threshold 0.05 --score_metric "cos" --max_speaker $max_spk \
			       --spt_est_thres 'None' \
			       --segment_file_input_path $dataDir/kaldi_xvectors/segments \
			       --spk_labels_out_path $expDir/SC/labels_estNumSpkr \
			       --sparse_search True || exit 1
		  mkdir -p $expDir/SC/clustering_estNumSpkr
		  
	  
	  else
		  python sc_utils/make_rttm.py $dataDir/kaldi_xvectors/segments \
		    $expDir/SC/labels_estNumSpkr $expDir/SC/clustering_estNumSpkr/rttm
		  python spectral_opt.py --affinity_score_file $expDir/SC/cos_scores/scores.scp \
			       --threshold 0.05 --score_metric "cos" --max_speaker $max_spk \
			       --spt_est_thres 'None' --reco2num_spk $dataDir/reco2num_spk \
			       --segment_file_input_path $dataDir/kaldi_xvectors/segments \
			       --spk_labels_out_path $expDir/SC/labels_estNumSpkr \
			       --sparse_search True || exit 1
		  mkdir -p $expDir/SC/clustering_estNumSpkr
		  python sc_utils/make_rttm.py $dataDir/kaldi_xvectors/segments \
		    $expDir/SC/labels_estNumSpkr $expDir/SC/clustering_estNumSpkr/rttm
	  fi
  fi
  cd ..

fi

# Evaluation
end=$(date +%s.%N) 
runtime=$(python -c "print(${end} - ${start})")
echo "Runtime was $runtime"
