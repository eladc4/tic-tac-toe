# tic-tac-toe

my take on https://medium.com/@carsten.friedrich/teaching-a-computer-to-play-tic-tac-toe-88feb838b5e3

## results for 100,000 games:

Player            | P1 Win | P2 Win  |  Draw
|:---|:---:|:---:|:---:|
Random - Random   |  58.4% |  29.2%  |  12.4%
MinMax - Random   |  99.4% |  0.0%  |  0.6%
Random - MinMax   |  0.0% |  80.1%  |  19.9%
Random - MinMax   |  0.0% |  80.4%  |  19.6%
MinMax - MinMax   |  0.0% |  0.0%  |  100.0%
TQ - Random   |  95.1% |  1.5%  |  3.4%
TQ - MinMax   |  0.0% |  5.7%  |  94.3%