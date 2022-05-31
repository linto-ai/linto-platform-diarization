#!/usr/bin/env python3
import os
import time
import logging
import os
import subprocess
class SpeakerDiarization:
    def __init__(self):
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)

        self.log = logging.getLogger(
            '__speaker-diarization-worker__' + '.' + __name__)

       
        self.maxNrSpeakers = 20  # If known, max nr of speakers in a sesssion in the database. This is to limit the effect of changes in very small meaningless eigenvalues values generating huge eigengaps

        
       

    def getSegments(self, frameshift, finalSegmentTable, finalClusteringTable, dur):
        numberOfSpeechFeatures = finalSegmentTable[-1, 2].astype(int)+1
        solutionVector = np.zeros([1, numberOfSpeechFeatures])
        for i in np.arange(np.size(finalSegmentTable, 0)):
            solutionVector[0, np.arange(
                finalSegmentTable[i, 1], finalSegmentTable[i, 2]+1).astype(int)] = finalClusteringTable[i]
        seg = np.empty([0, 3])
        solutionDiff = np.diff(solutionVector)[0]
        first = 0
        for i in np.arange(0, np.size(solutionDiff, 0)):
            if solutionDiff[i]:
                last = i+1
                seg1 = (first)*frameshift
                seg2 = (last-first)*frameshift
                seg3 = solutionVector[0, last-1]
                if seg.shape[0] != 0 and seg3 == seg[-1][2]:
                    seg[-1][1] += seg2
                elif seg3 and seg2 > 1:  # and seg2 > 0.1
                    seg = np.vstack((seg, [seg1, seg2, seg3]))
                first = i+1
        last = np.size(solutionVector, 1)
        seg1 = (first-1)*frameshift
        seg2 = (last-first+1)*frameshift
        seg3 = solutionVector[0, last-1]
        if seg3 == seg[-1][2]:
            seg[-1][1] += seg2
        elif seg3 and seg2 > 1:  # and seg2 > 0.1
            seg = np.vstack((seg, [seg1, seg2, seg3]))
        seg = np.vstack((seg, [dur, -1, -1]))
        seg[0][0] = 0.0
        return seg

    def format_response(self, segments: list) -> dict:
        #########################
        # Response format is
        #
        # {
        #   "speakers":[
        #       {
        #           "id":"spk1",
        #           "tot_dur":10.5,
        #           "nbr_segs":4
        #       },
        #       {
        #           "id":"spk2",
        #           "tot_dur":6.1,
        #           "nbr_segs":2
        #       }
        #   ],
        #   "segments":[
        #       {
        #           "seg_id":1,
        #           "spk_id":"spk1",
        #           "seg_begin":0,
        #           "seg_end":3.3,
        #       },
        #       {
        #           "seg_id":2,
        #           "spk_id":"spk2",
        #           "seg_begin":3.6,
        #           "seg_end":6.2,
        #       },
        #   ]
        # }
        #########################

        json = {}
        _segments = []
        _speakers = {}
        seg_id = 1
        spk_i = 1
        spk_i_dict = {}

        # Remove the last line of the segments.
        # It indicates the end of the file and segments.
        if segments[len(segments)-1][2] == -1:
            segments=segments[:len(segments)-1]

        for seg in segments:
            segment = {}
            segment['seg_id'] = seg_id

            # Ensure speaker id continuity and numbers speaker by order of appearance.
            if seg[2] not in spk_i_dict.keys():
                spk_i_dict[seg[2]] = spk_i
                spk_i += 1

            segment['spk_id'] = 'spk'+str(spk_i_dict[seg[2]])
            segment['seg_begin'] = float("{:.2f}".format(seg[0])) 
            segment['seg_end'] = float("{:.2f}".format(seg[0] + seg[1]))  
            
            if segment['spk_id'] not in _speakers:
                _speakers[segment['spk_id']] = {}
                _speakers[segment['spk_id']]['spk_id'] = segment['spk_id']
                _speakers[segment['spk_id']]['duration'] = float("{:.2f}".format(seg[1])) 
                _speakers[segment['spk_id']]['nbr_seg'] = 1
            else:
                _speakers[segment['spk_id']]['duration'] += seg[1]
                _speakers[segment['spk_id']]['nbr_seg'] += 1
                _speakers[segment['spk_id']]['duration'] = float("{:.2f}".format(_speakers[segment['spk_id']]['duration'])) 

            _segments.append(segment)
            seg_id += 1

        json['speakers'] = list(_speakers.values())
        json['segments'] = _segments
        return json

    
    
    def run(self, audioFile, number_speaker: int = None, max_speaker: int = None):
        try:
            start_time = time.time()
            
            if type(audioFile) is not str:
                filename = str(uuid.uuid4())
                file_path = "/tmp/"+filename
                audioFile.save(file_path)
            else:
                file_path = audioFile
            

            
            

            subprocess.check_call("./diarize.sh %s %s %s" % (str(wavDir), number_speaker, max_speaker), shell=True)
            
            
            bestClusteringID = pybk.getSpectralClustering(self.metric_clusteringSelection, 
                                                          finalClusteringTable, 
                                                          self.N_init, 
                                                          segmentBKTable, 
                                                          segmentCVTable, 
                                                          number_speaker, 
                                                          k, 
                                                          self.sigma, 
                                                          self.percentile, 
                                                          max_speaker if max_speaker is not None else self.maxNrSpeakers)+1

            if self.resegmentation and np.size(np.unique(bestClusteringID), 0) > 1:
                finalClusteringTableResegmentation, finalSegmentTable = pybk.performResegmentation(data,
                                                                                                   speechMapping, 
                                                                                                   mask, 
                                                                                                   bestClusteringID, 
                                                                                                   segmentTable, 
                                                                                                   self.modelSize, 
                                                                                                   self.nbIter, 
                                                                                                   self.smoothWin, 
                                                                                                   nSpeechFeatures)
                seg = self.getSegments(self.frame_shift_s, 
                                       finalSegmentTable, 
                                       np.squeeze(finalClusteringTableResegmentation), 
                                       duration)
            

            self.log.info("Speaker Diarization took %d[s] with a speed %0.2f[xRT]" %
                          (int(time.time() - start_time), float(int(time.time() - start_time)/duration)))
        except ValueError as v:
            self.log.error(v)
            raise ValueError('Speaker diarization failed during processing the speech signal')
        except Exception as e:
            self.log.error(e)
            raise Exception('Speaker diarization failed during processing the speech signal')
        else:
            return seg
