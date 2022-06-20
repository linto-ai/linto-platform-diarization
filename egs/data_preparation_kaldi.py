#!/usr/bin/env python3


import sys, os
import re
def prepare_data(src_dir, data_dir):
    
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
    prepare_data(src_dir, data_dir)
    return 0

if __name__=="__main__":
    main()
