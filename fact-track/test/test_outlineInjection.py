import sys
from outline_stubs import load_outline_stub

sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/core")
from outline import OutlineItem, Outline, generate_outline
from contradict_detector import ContradictDetector_StatusPersistence
from state_checker import OutlineItemStateChecker

sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/utils")
from gpt_api import load_model2classification, load_model2generation
from log_saver import LogSaver

sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/operation")
from outline_injection import generate_outline_withInjection

sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/analyze")
from injection_dataset import outline_analyze

def test_outlineInjection():
    logSaver = LogSaver("test_injection")
    outline = load_outline_stub()
    print("Before injection:")
    print(outline)
    outline = generate_outline_withInjection(logSaver, outline = outline, max_depth = 1, injection = "plot_fact", bandwidth = 3)
    print("After injection:")
    print(outline)
    logSaver.save()
    outline_analyze("test_injection")

if __name__ == "__main__":
    test_outlineInjection()