import sys
sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/core")
from outline import OutlineItem, Outline, generate_outline
from contradict_detector import ContradictDetector_StatusPersistence


from outline_stubs import load_outline

def test_getPrompt(outline):
    print(outline.get_prompt("2.3", prompt_method =  "simple"))
    print("#"*200)
    print(outline.get_prompt("2.3", prompt_method =  "detail"))
    print("#"*200)
    print(outline.get_prompt("2.3", prompt_method =  "full"))


if __name__ == "__main__":
    outline = load_outline()
    test_getPrompt(outline)