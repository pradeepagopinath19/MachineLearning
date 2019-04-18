class Rectangle():
    def __init__(self, top_left, bottom_right):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.bottom_left = [self.top_left[0], self.bottom_right[1]]
        self.top_right = [self.bottom_right[0], self.top_left[1]]

        # Computations
        self.width = self.compute_width()
        self.height = self.compute_height()

    def compute_area(self):
        return self.width * self.height

    def compute_width(self):
        return self.bottom_right[0] - self.bottom_left[0]

    def compute_height(self):
        return self.bottom_left[1] - self.top_left[1]
