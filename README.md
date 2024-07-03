# README

## File Structure
* `fact-track/core`: three core object for FactTrack, which is, `outline`, `contradict detector` and `state checker` respectively.
* `fact-track/operation`: support three types of operations, `outline generation`, `outline detection` and `outline injection`.
* `fact-track/utils`: LLM apis and saving the log
* `fact-track/test`: test code
* `fact-track/analyze`: the code for cleanning the structure after the detection
* `fact-track/autoAnno`: the code for baseline
* `fact-track/oddAnno`: the code for metric session
* `fact-track/nli`: the code to fine tuning NLI model for fact-level contradict classification

## How to use
### Install the requiremnet 
```bash
conda env create -f env.yml
```
After that, download the files (premise and model) on the root folder, 
### Generate a story outline
Under `fact-track/operation`
```bash
python outline_generation.py --begin <begin number> --end <end number>
```
### Detect Contradictions on a story outline
```bash
python outline_detection.py --begin <begin number> --end <end number>
```
### Generate a story outline with rewriting
```bash
python outline_injection.py
```
## Files to download
* [Premise](https://drive.google.com/file/d/1X-MGuToTBVB_wBk0wFgCFIjPBXaHt8hM/view?usp=sharing)
* [Model](https://drive.google.com/file/d/1W0oe1KkaXUlqQZd5uc7s0AsVkPYoaJjl/view?usp=sharing)
* [Data Generated](https://drive.google.com/file/d/150qsMNI2lmLrsFIfiGwNyxqwZdiJXS5O/view?usp=sharing)
* [Data Analisis](https://drive.google.com/file/d/1XQXv3fGscLEp1dyvNrIqrB6H4miJLWZi/view?usp=sharing)
## Cite this work
```latex
@inproceedings{,
  title={},
  author={},
  booktitle={},
  year={},
  organization={}
}
```

