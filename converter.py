import numpy as np
from sklearn import linear_model

class GradeConverter():

    def __init__(self):
        self.us = np.array([2, 2.33, 2.67, 3, 3.33, 3.67, 4], dtype=float).reshape(-1, 1)
        self.uk = np.array([50, 53.3, 56.7, 60, 63.3, 66.7, 70], dtype=float).reshape(-1, 1)

    def _model(self, lang_from = 'uk', lang_to = 'us'):
        # Create linear regression object
        regr = linear_model.LinearRegression()

        try:
            x = getattr(self, lang_from)
        except AttributeError:
            raise NotImplementedError(
                "Class `{}` does not implement `{}`".format(
                    self.__class__.__name__, lang_from))
        try:
            y = getattr(self, lang_to)
        except AttributeError:
            raise NotImplementedError(
                "Class `{}` does not implement `{}`".format(
                    self.__class__.__name__, lang_to))


        # Train the model using the training sets
        regr.fit(x, y)

        return regr

    @staticmethod
    def cap(lang_to,x):
        if lang_to == 'us':
            return np.array([4 if (i >= 4) else i[0] for i in x])
        else:
            return x

    def convert(self,lang_from,lang_to,x):
        regr = self._model(lang_from,lang_to)
        conv_grades = regr.predict(x)
        conv_grades = self.cap(lang_to,conv_grades)
        avg_grade = np.average(conv_grades)
        return conv_grades, avg_grade



if __name__=='__main__':
    year_1 = [75, 92, 67, 66.7, 78, 69, 69, 83, 75]
    year_2 = [86, 82, 73, 74, 59, 79, 61, 93]

    all_years = []
    all_years.extend(year_1)
    all_years.extend(year_2)

    my_grades = np.array(
       all_years
    ).reshape(-1, 1)

    gc = GradeConverter()

    convs,avg = gc.convert('uk','us',my_grades)

    print('Your converted grades:\n\t{}\n\nYour average grade:\n\t{}'.format(
        convs,format(avg,'.2f')))
