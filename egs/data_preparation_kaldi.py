#!/usr/bin/env python3

# This script is called by local/make_dihard_2018_eval.sh, and it creates the
# necessary files for DIHARD 2018 evaluation directory.

import sys, os
import re
def prepare_dihard_2018_eval(src_dir, data_dir):
    #wavscp_fi = open(data_dir + "/wav.scp" , 'w')
    #utt2spk_fi = open(data_dir + "/utt2spk" , 'w')
    #segments_fi = open(data_dir + "/segments" , 'w')
    #rttm_fi = open(data_dir + "/rttm" , 'w')
    #reco2num_spk_fi = open(data_dir + "/reco2num_spk" , 'w')
    with open(os.path.join(data_dir, "wav.scp"), "w") as fout:
		    for speaker in sorted(os.listdir(src_dir)):
		        print(speaker)
		        absPath = os.path.join(src_dir, speaker)
		        fout.write(
		                        "{} {}\n".format(speaker[:-4], absPath)
		                        )
    with open(os.path.join(data_dir, "utt2spk"), "w") as fout1:
		    for speaker in sorted(os.listdir(src_dir)):
		        
		        fout1.write(
		                        "{} {}\n".format(speaker[:-4], speaker[:-4])
		                        )
		       		       

    return 0

def main():
    src_dir = sys.argv[1]
    data_dir = sys.argv[2]
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    prepare_dihard_2018_eval(src_dir, data_dir)
    return 0

if __name__=="__main__":
    main()
