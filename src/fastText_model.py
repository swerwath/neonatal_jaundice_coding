import fastText
import os
import pickle
from random import shuffle

class ClassificationModel():
    def __init__(self, path=None):
        if path is not None:
            self.model = fastText.load_model(os.path.join(path, "ft.bin"))
            return

    def train(self, data, labels, dim=100, ng=2, epoch=10):
        if len(data) != len(labels):
            raise ValueError("Length of data (" + str(len(data)) + ") does not match length of labels (" + str(len(labels)) + ")" )

        # Convert training data into strings for fastText
        mapped_report_strs = []
        for i in range(len(data)):
            report_string = data[i].replace("\n", " ")
            label = " __label__" + str(labels[i])
            mapped_report_strs.append(report_string + label)
        shuffle(mapped_report_strs)

        # Write strings to a temp file
        train_path = "./MODEL_TRAIN_TEMP.bin"
        with open(train_path, 'w') as outfile:
            for mrs in mapped_report_strs:
                outfile.write(mrs)
                outfile.write("\n")

        # Train fastText model
        self.model = fastText.train_supervised(train_path, dim=dim, epoch=epoch, thread=4, wordNgrams=ng)

        # Delete temp file
        os.remove(train_path)

    def predict(self, report):
        prediction = self.model.predict(report)
        conf = 0.5 - (prediction[1][0] / 2) if prediction[0][0] == '__label__0' else 0.5 + (prediction[1][0] / 2)
        return conf

    def save_model(self, path):
        os.mkdir(path)
        self.model.save_model(os.path.join(path, "ft.bin"))

    def get_words(self):
        return self.model.get_words()
