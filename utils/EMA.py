class EMA:
    def __init__(self, alpha, s0=None):
      self.alpha = alpha
      self.s = s0
  
    def update(self, x):
      if self.s is None:
          self.s = x
      else:
          self.s += self.alpha * (x - self.s)
    
    def value(self):
      if self.s is None:
          raise ValueError('No observations yet')
      return self.s