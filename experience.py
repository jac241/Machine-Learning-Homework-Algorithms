class Experience:
    def __init__(self, label, y, x):
        self.label = label
        self.y = y
        self.x = x
        self.n = len(x)

    def __str__(self):
        result = '%s: %s - ' % (self.label, self.y)
        for feature in self.x:
            result += str(feature) + ' '
        return result
