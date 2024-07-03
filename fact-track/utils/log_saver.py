import os
BASE_PATH = os.environ["BASE_PATH"]

import sys
sys.path.append(f"{BASE_PATH}/fact-track/fact-track/core")
from outline import OutlineItem, Outline, generate_outline
from contradict_detector import ContradictDetector_StatusPersistence
from state_checker import OutlineItemStateChecker

sys.path.append(f"{BASE_PATH}/fact-track/fact-track/utils")
from gpt_api import load_model2classification, load_model2generation

import os
class LogSaver(object):
    def __init__(self, metaname = "sample"):
        self.metaname = f"{BASE_PATH}/fact-track/data/" + metaname
        if not os.path.exists(self.metaname):
            os.makedirs(self.metaname)
        self.model_inGeneration = None
        self.model_inDecomposition = None
        self.model_inDetection = None
        self.model_inRewrite = None
        self.model_summary = None
        self.outline = None
        self.detector = None
        self.stateChecker_dict = {}

    def add_model(self, model_inGeneration = None, model_inDecomposition = None,
                  model_inDetection = None, model_inRewrite = None):
        if model_inGeneration != None:
            self.model_inGeneration = model_inGeneration
        if model_inDecomposition != None:
            self.model_inDecomposition = model_inDecomposition
        if model_inDetection != None:
            self.model_inDetection = model_inDetection
        if model_inRewrite != None:
            self.model_inRewrite = model_inRewrite

    def add_outline(self, outline):
        if self.outline is not None:
            print("Warning: outline is not None, it will be overwrited.")
        self.outline = outline

    def add_detector(self, detector):
        if self.detector is not None:
            print("Warning: detector is not None, it will be overwrited.")
        self.detector = detector

    def add_stateChecker(self, idx, stateChecker):
        # idx = stateChecker.outlineItem.idx
        if idx in self.stateChecker_dict:
            print("Warning: stateChecker is not None, it will be overwrited.")
        self.stateChecker_dict[idx] = stateChecker

    def save_model_logs(self):
        dir = self.metaname + "/data"
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = dir + "/model_logs.txt" # TODO: need to specify
        # clean the file
        with open(path, "w") as f:
            f.write("")
        if self.model_inGeneration is not None:
            with open(path, "a") as f:
                if type(self.model_inGeneration) == str:
                    f.write("Model in Generation:"+self.model_inGeneration)
                else:
                    f.write("Model in Generation:"+str(self.model_inGeneration.summarize))
            print("model_inGeneration saved.")
        if self.model_inDecomposition is not None:
            with open(path, "a") as f:
                if type(self.model_inDecomposition) == str:
                    f.write("Model in Decomposition:"+self.model_inDecomposition)
                else:
                    f.write("Model in Decomposition:"+str(self.model_inDecomposition.summarize))
            print("model_inDecomposition saved.")
        if self.model_inDetection is not None:
            with open(path, "a") as f:
                if type(self.model_inDetection) == str:
                    f.write("Model in Detection:"+self.model_inDetection)
                else:
                    f.write("Model in Detection:"+str(self.model_inDetection.summarize))
            print("model_inDetection saved.")
        if self.model_inRewrite is not None:
            with open(path, "a") as f:
                if type(self.model_inRewrite) == str:
                    f.write("Model in Rewrite:"+self.model_inRewrite)
                else:
                    f.write("Model in Rewrite:"+str(self.model_inRewrite.summarize))
            print("model_inRewrite saved.")

    def save_outline(self):
        if self.outline is None:
            return
        dir = self.metaname + "/object"
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = dir + "/outline.pkl"
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.outline, f)

        # todo: save the text format outline.
        # dir = self.metaname + "/data"
        # if not os.path.exists(dir):
        #     os.makedirs(dir)
        # path = dir + "/outline.txt"
        # import pickle
        # with open(path, "wb") as f:
        #     pickle.dump(self.outline, f)
        print("outline saved.")

    def save_detector(self):
        if self.detector is None:
            return
        dir = self.metaname + "/object"
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = dir + "/detector.pkl" # TODO: need to specify
        import pickle
        self.detector.model_decompose = None
        self.detector.model_contradict = None
        with open(path, "wb") as f:
            pickle.dump(self.detector, f)
        print("detector saved.")

    def save_stateChecker(self):
        if len(self.stateChecker_dict) == 0:
            return
        dir = self.metaname + "/object"
        if not os.path.exists(dir):
            os.makedirs(dir)
        # clean the dict, remove the object inside:
        for key in self.stateChecker_dict:
            self.stateChecker_dict[key].contradictDetector = None
            self.stateChecker_dict[key].model_rewrite = None
        # directly save the stateChecker_dict by pickle
        path = dir + "/stateChecker_dict.pkl" # TODO: need to specify
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.stateChecker_dict, f)
        print("stateChecker_dict saved.")

    def save(self):
        if self.detector != None:
            self.detector.end_log()
        self.save_model_logs()
        self.save_outline()
        self.save_detector()
        self.save_stateChecker()

    def remove(self):
        import shutil
        shutil.rmtree(self.metaname)
        print("removed.")

    def load(self):
        # load from different path, then return the export
        print(self.metaname)
        dir_data = self.metaname + "/data"
        dir_object = self.metaname + "/object"
        if os.path.exists(dir_data):
            self.model_summary = []
            path = dir_data + "/model_logs.txt"
            with open(path, "r") as f:
                for line in f.readlines():
                    self.model_summary.append(line)
            print("model_logs.txt loaded.")
        if os.path.exists(dir_object):
            import pickle
            try:
                path = dir_object + "/outline.pkl"
                with open(path, "rb") as f:
                    self.outline = pickle.load(f)
                print("outline.pkl loaded.")
            except:
                print("outline.pkl not found.")

            try:
                path = dir_object + "/detector.pkl"
                print(path)
                with open(path, "rb") as f:
                    self.detector = pickle.load(f)
                print("detector.pkl loaded.")
            except:
                print("detector.pkl not found.")
            try:
                path = dir_object + "/stateChecker_dict.pkl"
                with open(path, "rb") as f:
                    self.stateChecker_dict = pickle.load(f)
                print("stateChecker_dict.pkl loaded.")
            except:
                print("stateChecker_dict.pkl not found.")
        # print(self.export().keys())
        return self.export()

    def export(self):
        # return a dict for all current elements
        export_dict = {}
        if self.model_inGeneration is not None:
            export_dict["model_inGeneration"] = self.model_inGeneration.summarize
        if self.model_inDecomposition is not None:
            export_dict["model_inDecomposition"] = self.model_inDecomposition.summarize
        if self.model_inDetection is not None:
            export_dict["model_inDetection"] = self.model_inDetection.summarize
        if self.model_inRewrite is not None:
            export_dict["model_inRewrite"] = self.model_inRewrite.summarize
        if self.model_summary is not None:
            export_dict["model_summary"] = self.model_summary
        if self.detector is not None:
            export_dict["detector"] = self.detector
        if self.outline is not None:
            export_dict["outline"] = self.outline
        if self.stateChecker_dict is not None:
            export_dict["stateChecker_dict"] = self.stateChecker_dict
        return export_dict

if __name__ == "__main__":
    pass