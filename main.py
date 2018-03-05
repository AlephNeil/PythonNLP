import nltk
from nameparser.parser import HumanName as HN
import os

def human_names(t):
    tokens = nltk.word_tokenize(t)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary=False)
    for subtree in sentt.subtrees(filter=lambda t: t.label()=='PERSON'):
        ls = subtree.leaves()
        if len(ls) > 1:
            yield ' '.join(leaf[0] for leaf in ls)
         
text = """
Some economists have responded positively to Bitcoin, including 
Francois R. Velde, senior economist of the Federal Reserve in Chicago 
who described it as "an elegant solution to the problem of creating a 
digital currency." In November 2013 Richard Branson announced that 
Virgin Galactic would accept Bitcoin as payment, saying that he had invested 
in Bitcoin and found it "fascinating how a whole new global currency 
has been created", encouraging others to also invest in Bitcoin.
Other economists commenting on Bitcoin have been critical. 
Economist Paul Krugman has suggested that the structure of the currency 
incentivizes hoarding and that its value derives from the expectation that 
others will accept it as payment. Economist Larry Summers has expressed 
a "wait and see" attitude when it comes to Bitcoin. Nick Colas, a market 
strategist for ConvergEx Group, has remarked on the effect of increasing 
use of Bitcoin and its restricted supply, noting, "When incremental 
adoption meets relatively fixed supply, it should be no surprise that 
prices go up. And that’s exactly what is happening to BTC prices."
"""

if __name__ == '__main__':
    for name in set(human_names(text)):
        print(f'{HN(name).last}, {HN(name).first}')