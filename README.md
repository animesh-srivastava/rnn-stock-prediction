# Stock price prediction using RNN

Thanks to @aranroussi for [yfinance module](https://github.com/ranaroussi/yfinance)

you can install it on anaconda using the following
<pre><code>conda install -c ranaroussi yfinance</code></pre>
This model uses tensorflow 1.15 GPU for the computation, but it can easily be run on tensorflow 1.x on CPU
I still have to check for compatibility with tensorflow 2.x

This code uses past 80 days' data to predict the stock price on 81st day using Long Short Term Memory modules from Keras module of Tensorflow
