from jinja2 import *
import numpy as np
from scipy.stats import zipf
import sys

## Counter object for numbering the rules
class _Counter(object):
  def __init__(self, start_value=1):
    self.value=start_value

  def current(self):
    return self.value

  def next(self):
    v=self.value
    self.value+=1
    return v
    

## Assign Zipfian distribution to vocab
def normalize(probs):
    ### AK: Why is this not simply probs/sum(probs)?
    ### The code below does not maintain the ratios between the different probabilities.
    leftover_prob = 1-sum(probs)
    probs = probs + leftover_prob/len(probs)
    return probs

def generate_vocab_probabilities(words):
    a = 1.4
    k = np.array(list(range(1,101)))
    probs = zipf.pmf(np.array(range(1,len(words)+1)), a)
    probs = normalize(probs)
    return zip(words, probs)






## Process the templates
env = Environment(loader=FileSystemLoader("."))
env.globals['counter'] = _Counter
env.filters['zipf'] = generate_vocab_probabilities

template = env.get_template(sys.argv[1])
print(template.render())


