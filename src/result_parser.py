import argparse
import csv

import numpy as np
from enum import Enum
from matplotlib import pyplot as plt

import shared_variables
from shared_variables import folds


class ResultParser:
    class HighlightTypes(Enum):
        ROW_SCORE = 0
        IMPROVEMENT_SCORE = 1

    class ColumnTypes(Enum):
        CF = 0
        R = 1

    _all_log_names = [
        '10x2_1W',
        '10x2_3W',
        '10x2_1S',
        '10x2_3S',
        '10x5_1W',
        '10x5_3W',
        '10x5_1S',
        '10x5_3S',
        '10x20_1W',
        '10x20_3W',
        '10x20_1S',
        '10x20_3S',
        '5x5_1W',
        '5x5_3W',
        '5x5_1S',
        '5x5_3S',
        '50x5_1W',
        '50x5_3W',
        '50x5_1S',
        '50x5_3S'
        'BPI2017_W',
        'BPI2017_S'
    ]

    _headers = ['B1_CF', 'B1_R', 'B1_CF', 'B1_R', 'B2_CF', 'B2_R', 'B2_CF', 'B2_R', 'NEW_CF', 'NEW_R']
    _metrics = ['baseline', 'LTL', 'declare']

    _model_types = ['CF', 'CFR']

    # <editor-fold desc="reference_table">
    _reference_table = np.array(
        [[0.693, 0.742, 0.511, 0.658, 0.804, 0.729, 0.390, 0.765, 0.727, 0.784, 0.827, 0.849, 0.618,
          0.729, 0.727, 0.409, 0.553, 0.314, 0.231, 0.623, 0.000, 0.000],
         [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
          0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
         [0.643, 0.806, 0.258, 0.629, 0.810, 0.723, 0.695, 0.831, 0.864, 0.761, 0.784, 0.774, 0.645,
          0.749, 0.655, 0.806, 0.735, 0.506, 0.466, 0.802, 0.000, 0.000],
         [0.674, 0.682, 0.803, 0.793, 0.800, 0.605, 0.700, 0.741, 0.579, 0.583, 0.608, 0.566, 0.677,
          0.574, 0.642, 0.644, 0.642, 0.308, 0.377, 0.803, 0.000, 0.000],
         [0.649, 0.742, 0.682, 0.682, 0.803, 0.719, 0.479, 0.769, 0.909, 0.784, 0.884, 0.556, 0.616,
          0.729, 0.726, 0.413, 0.746, 0.674, 0.874, 0.732, 0.000, 0.000],
         [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
          0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
         [0.665, 0.806, 0.536, 0.662, 0.813, 0.723, 0.720, 0.846, 0.855, 0.723, 0.884, 0.620, 0.639,
          0.750, 0.691, 0.806, 0.707, 0.819, 0.825, 0.789, 0.000, 0.000],
         [0.640, 0.682, 0.760, 0.826, 0.800, 0.605, 0.717, 0.761, 0.579, 0.571, 0.627, 0.449, 0.684,
          0.574, 0.650, 0.643, 0.561, 0.514, 0.623, 0.799, 0.000, 0.000],
         [0.646, 0.803, 0.695, 0.667, 0.780, 0.831, 0.802, 0.849, 0.864, 0.838, 0.884, 0.000, 0.601,
          0.754, 0.718, 0.846, 0.697, 0.735, 0.870, 0.811, 0.000, 0.000],
         [0.649, 0.679, 0.763, 0.833, 0.790, 0.824, 0.792, 0.761, 0.575, 0.738, 0.704, 0.000, 0.785,
          0.758, 0.763, 0.679, 0.687, 0.742, 0.875, 0.815, 0.000, 0.000]]).T

    # </editor-fold>

    _column_colors = {
        ColumnTypes.CF: 'C0C0C0',
        ColumnTypes.R: 'E8E8E8'
    }

    def __init__(self, log_names):
        self._log_names = log_names

    @staticmethod
    def _parse_log(filepath, two_predictions=False):
        label_1 = 'Damerau-Levenshtein'
        label_2 = 'Damerau-Levenshtein Resource'

        if two_predictions:
            scores = [[], []]
        else:
            scores = [[]]

        with open(filepath, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='|')
            csv_headers = next(csv_reader)

            for row in csv_reader:
                score_1 = float(row[csv_headers.index(label_1)])
                scores[0].append(score_1)

                if two_predictions:
                    score_2 = float(row[csv_headers.index(label_2)])
                    scores[1].append(score_2)

        scores = np.mean(np.array(scores), -1)
        return scores

    def _populate_table(self, table, scores, log_name, metric, model_type):
        row = self._log_names.index(log_name)
        column = ResultParser._metrics.index(metric) * len(
            self._model_types) * 2 + self._model_types.index(model_type) * len(
            self._model_types)

        if metric == 'declare':
            column -= 2

        table[row, column] = scores[0]
        if scores.shape[0] == 2:
            table[row, column + 1] = scores[1]

    @staticmethod
    def _print_latex_table_header():
        print('\\begin{table}[!hbt]')
        print('\\centering')
        print('\\begin{tabular}{|l||c|c|c|c||c|c|c|c||c|c|}')
        print('\\hline')
        print('\\textit{Method $\\rightarrow$} & '
              '\\multicolumn{4}{|c||}{\\textbf{baseline}} & '
              '\\multicolumn{4}{|c||}{\\textbf{LTL}} & '
              '\\multicolumn{2}{|c|}{\\textbf{declare}} \\\\')
        print('\\hline')
        print('\\textit{Predictor $\\rightarrow$} & '
              '\\multicolumn{2}{|c|}{\\textbf{CF}} & '
              '\\multicolumn{2}{|c||}{\\textbf{CF+R}} & '
              '\\multicolumn{2}{|c|}{\\textbf{CF}} & '
              '\\multicolumn{2}{|c||}{\\textbf{CF+R}} & '
              '\\multicolumn{2}{|c|}{\\textbf{CF+R}} \\\\')
        print('\\hline')
        print('\\textit{Predicand $\\rightarrow$} & '
              '\\textbf{CF} & \\textbf{R} & \\textbf{CF} & '
              '\\textbf{R} & \\textbf{CF} & \\textbf{R} & '
              '\\textbf{CF} & \\textbf{R} & \\textbf{CF} & '
              '\\textbf{R}\\\\')
        print('\\hline\\hline')

    @staticmethod
    def _print_latex_table_footer(table_caption, table_label):
        print('\\hline')
        print('\\end{tabular}')
        print('\\caption{' + table_caption + '}')
        print('\\label{' + table_label + '}')
        print('\\end{table}')

    @staticmethod
    def _print_score(score, reference_score, highlight_type, column_type):
        column_color = ResultParser._column_colors[column_type]

        if highlight_type == ResultParser.HighlightTypes.ROW_SCORE:
            if score == reference_score:
                print('\\cellcolor[HTML]{%s}\\textbf{%.3f}' % (column_color, score)),
            else:
                print('%.3f' % score),
        elif highlight_type == ResultParser.HighlightTypes.IMPROVEMENT_SCORE:
            if score > 0:
                print('\\cellcolor[HTML]{%s}\\textbf{%.3f}' % (column_color, score)),
            else:
                print('%.3f' % score),
        else:
            print('%.3f' % score),

    def _print_latex_table(self, populated_table, highlight_type, table_caption, table_label):
        cf_maximums = np.max(populated_table[:, 0::2], 1)
        r_maximums = np.max(populated_table[:, 1::2], 1)

        self._print_latex_table_header()
        for i, log_name in enumerate(self._log_names):
            print(log_name.replace('_', '\_') + ' & '),
            for j, score in enumerate(populated_table[i]):
                if j % 2 == 0:
                    self._print_score(score, cf_maximums[i], highlight_type, ResultParser.ColumnTypes.CF)
                else:
                    self._print_score(score, r_maximums[i], highlight_type, ResultParser.ColumnTypes.R)

                if j != populated_table.shape[1] - 1:
                    print('&'),
                else:
                    print('\\\\')

        self._print_latex_table_footer(table_caption, table_label)

    def _show_comparison_image(self, target_table, reference_table):
        improvement_percentage = (1.0 * np.count_nonzero(target_table > reference_table) / np.count_nonzero(
            target_table)) * 100.0

        populated_indexes = np.where(target_table > 0)
        mean_improvement = float(np.mean(target_table[populated_indexes] - reference_table[populated_indexes]))
        sum_improvement = float(np.sum(target_table[populated_indexes] - reference_table[populated_indexes]))

        plt.subplots(1, 2)
        plt.subplot(1, 2, 1)
        plt.title('better performance (%.2f' % improvement_percentage + '%)')
        binary_image = np.zeros(target_table.shape)
        binary_image[target_table == 0] = -1.0
        binary_image[target_table > reference_table] = 1.0
        plt.imshow(binary_image, cmap='hot')
        plt.yticks(range(len(self._log_names)), self._log_names)
        plt.xticks(range(len(self._headers)), self._headers, rotation=90)

        plt.subplot(1, 2, 2)
        plt.title('comparison (mean:%.2f, sum:%.2f' % (mean_improvement, sum_improvement) + ')')
        gradient_image = target_table - reference_table
        gradient_image[target_table == 0] = 0
        plt.imshow(gradient_image, vmin=-1, vmax=1,
                   cmap=plt.cm.seismic)
        plt.yticks(range(len(self._log_names)), self._log_names)
        plt.xticks(range(len(self._headers)), self._headers, rotation=90)
        plt.tight_layout()
        plt.show()

    def _load_table(self, folderpath):
        if folderpath == 'reference':
            return self._reference_table[sorted(self._all_log_names.index(i) for i in self._log_names)]
        elif folderpath == 'zeros':
            return np.zeros((len(self._log_names), len(self._metrics) * len(self._model_types) * 2 - 2))
        else:
            table_folds = []
            for fold in range(folds):
                fold_table = np.zeros(
                    (len(self._log_names), len(self._metrics) * len(self._model_types) * 2 - 2))

                for log_name in self._log_names:
                    for metric in self._metrics:
                        for model_type in self._model_types:
                            if metric == 'declare' and model_type == 'CF':
                                continue
                            filepath = folderpath + str(
                                fold) + '/results/' + metric + '/' + log_name + '_' + model_type + '.csv'
                            scores = self._parse_log(filepath, model_type == 'CFR')
                            self._populate_table(fold_table, scores, log_name, metric, model_type)
                table_folds.append(fold_table)
            populated_table = np.mean(table_folds, axis=0)
            return populated_table

    def compare_results(self, target, reference='zeros', highlight_type=HighlightTypes.ROW_SCORE, table_caption='',
                        table_label=''):
        target_table = self._load_table(target)
        reference_table = self._load_table(reference)

        # self._show_comparison_image(target_table, reference_table)
        self._print_latex_table(target_table - reference_table, highlight_type, table_caption, table_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', help='input logs')
    parser.add_argument('--target_model', default='new_model', help='target model name')
    parser.add_argument('--reference_model', default='old_model', help='reference model name')
    parser.add_argument('--table_caption', default='', help='final latex caption')
    parser.add_argument('--table_label', default='', help='final latex label')
    args = parser.parse_args()

    result_parser = ResultParser(args.logs.replace('[', '').replace(']', '').split(','))

    models_dict = {
        'old_model': shared_variables.outputs_folder + 'old_model/',
        'new_model': shared_variables.outputs_folder + 'new_model/',
        'new_model_2': shared_variables.outputs_folder + 'new_model_2/',
        'reference': 'reference',
        'zeros': 'zeros'
    }

    result_parser.compare_results(models_dict[args.target_model], reference=models_dict[args.reference_model],
                                  table_caption=args.table_caption, table_label=args.table_label)
