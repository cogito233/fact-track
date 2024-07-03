import sys
sys.path.append("/home/yangk/zhiheng/develop_codeversion/fact-track/fact-track/core")
from outline import OutlineItem, Outline, generate_outline
from contradict_detector import ContradictDetector_StatusPersistence

def load_outline():
    # this function is to load a depth-2 outline
    path = "/home/yangk/zhiheng/develop_codeversion/fact-track/cache_data/outline/sample.pkl"
    import pickle
    with open(path, "rb") as f:
        outline = pickle.load(f)
    return outline

def load_outline_stub():
    plot_point_text_1 = "Max purchases the book and leaves the antique store."
    plot_point_text_2 = "The Antique Store Owner shows Max a strange book that catches his eye."
    sonOutline1 = f"""1
Main plot: {plot_point_text_1}
Characters: Max
Begin Event: Max enters the antique store.
End Event: Max purchases the book and leaves the antique store.
"""
    sonOutline2 = f"""2
Main plot: {plot_point_text_2}
Characters: Max, The Antique Store Owner
Begin Event: The Antique Store Owner shows Max a strange book that catches his eye.
End Event: Max purchases the book and leaves the antique store.
"""
    outline = Outline([sonOutline1, sonOutline2], premise = "This is a premise.")
    return outline

if __name__ == "__main__":
    outline = load_outline_stub()
    print(outline)