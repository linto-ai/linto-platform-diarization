# 1.0.2
- Changed: Diarization parameters to reduce computation time
- Fixed: Speaker id shoulds now be continuous and numeroted by order of appearance. 
- Removed: Lot of unused pybk code

# 1.0.1
- Added max_speaker field for HTTP and celery requests.
- Fixed max_speaker not being properly considered during clustering.
- Removed PyBK as a submodule in favor of a hard copy.
- Updated README and API documentation.

# 1.0.0
- Diarization service bases on PyBK.
- Celery connectivity
- HTTP API