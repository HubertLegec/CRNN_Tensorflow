
def batch_levenshtein_distance(predictions, labels) -> list:
    distances = []
    for index, gt_label in enumerate(labels):
        pred = predictions[index]
        distance = levenshtein_distance(pred, gt_label)
        distances.append(distance)
    return distances


def levenshtein_distance(s: str, t: str):
    """
        Calculates the Levenshtein distance between the strings s and t.
        For all i and j, dist[i,j] will contain the Levenshtein
        distance between the first i characters of s and the first j characters of t

        source: https://www.python-course.eu/levenshtein_distance.php
    """
    rows = len(s) + 1
    cols = len(t) + 1

    dist = [[0 for _ in range(cols)] for _ in range(rows)]
    # source prefixes can be transformed into empty strings by deletions:
    for row in range(1, rows):
        dist[row][0] = row
    # target prefixes can be created from an empty source string by inserting the characters
    for col in range(1, cols):
        dist[0][col] = col

    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row - 1][col] + 1,
                                 dist[row][col - 1] + 1,
                                 dist[row - 1][col - 1] + cost)  # substitution

    return dist[rows - 1][cols - 1]
