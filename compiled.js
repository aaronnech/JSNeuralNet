function Layer(a,b){this.previousLayer_=b;this.neurons_=[];for(var d=0;d<a;d++){var c=null;b&&(c=b.getNeurons());c=new Neuron(c,!0);this.neurons_.push(c)}}Layer.prototype.getMeanSquaredError=function(){for(var a=0,b=0;b<this.neurons_.length;b++)a+=Math.pow(this.neurons_[b].getError(),2);return a/this.neurons_.length};Layer.prototype.setBias=function(a){for(var b=0;b<this.neurons_.length;b++)this.neurons_[b].setBias(a)};
Layer.prototype.propagateError=function(a){a=a.getNeurons();for(var b=0;b<this.neurons_.length;b++){for(var d=this.neurons_[b],c=d.getOutput(),e=0,f=0;f<a.length;f++)var g=a[f].getErrorDelta(),h=a[f].getWeight(b),e=e+g*h;d.setError(e);d.setErrorDelta(e*c*(1-c))}};Layer.prototype.updateWeights=function(a,b){for(var d=0;d<this.neurons_.length;d++)this.neurons_[d].updateWeights(a,b)};Layer.prototype.synapse=function(){for(var a=0;a<this.neurons_.length;a++)this.neurons_[a].synapse()};
Layer.prototype.getNeurons=function(){return this.neurons_};Layer.prototype.setPreviousLayer=function(a){for(var b=0;b<this.neurons_.length;b++)this.neurons_[b].setInputs(a.getNeurons())};function NeuralNetwork(a,b,d,c){this.inputCount_=a;this.outputCount_=b;this.learningRate_=d;this.momentum_=c;this.inputLayer_=new Layer(this.inputCount_,null);this.inputLayer_.setBias(0);this.outputLayer_=new Layer(this.outputCount_,this.inputLayer_);this.hiddenLayers_=[]}NeuralNetwork.prototype.addLayer=function(a){var b=this.inputLayer_;0<this.hiddenLayers_.length&&(b=this.hiddenLayers_[this.hiddenLayers_.length-1]);a=new Layer(a,b);this.hiddenLayers_.push(a);this.outputLayer_.setPreviousLayer(a)};
NeuralNetwork.prototype.forwardFeed_=function(){this.inputLayer_.synapse();for(var a=0;a<this.hiddenLayers_.length;a++)this.hiddenLayers_[a].synapse();this.outputLayer_.synapse()};NeuralNetwork.prototype.calculateError_=function(a,b){for(var d=this.outputLayer_.getNeurons(),c=0;c<this.outputCount_;c++){var e=b[c]-a[c];d[c].setError(e);d[c].setErrorDelta(e*a[c]*(1-a[c]))}for(c=this.hiddenLayers_.length-1;0<=c;c--)d=null,d=c==this.hiddenLayers_.length-1?this.outputLayer_:this.hiddenLayers_[c+1],this.hiddenLayers_[c].propagateError(d)};
NeuralNetwork.prototype.updateWeights_=function(){this.outputLayer_.updateWeights(this.learningRate_,this.momentum_);for(var a=0;a<this.hiddenLayers_.length;a++)this.hiddenLayers_[a].updateWeights(this.learningRate_,this.momentum_)};NeuralNetwork.prototype.backwardPropagate_=function(a,b){this.calculateError_(a,b);this.updateWeights_()};
NeuralNetwork.prototype.trainSet=function(a,b,d,c){if(a.length!=b.length)throw"Invalid training set";for(var e=1,f=0;f<c&&e>d;){for(var g=e=0;g<a.length;g++)e+=this.train(a[g],b[g]);e/=a.length;f++}};
NeuralNetwork.prototype.train=function(a,b){if(a.length!=this.inputCount_||b.length!=this.outputCount_)throw"Invalid training data or expected output dimensions.";for(var d=this.inputLayer_.getNeurons(),c=0;c<this.inputCount_;c++)d[c].setBias(a[c]);this.forwardFeed_();this.backwardPropagate_(this.getOutputs(),b);return this.outputLayer_.getMeanSquaredError()};
NeuralNetwork.prototype.predict=function(a){if(a.length!=this.inputCount_)throw"Invalid data dimensions.";for(var b=this.inputLayer_.getNeurons(),d=0;d<this.inputCount_;d++)b[d].setBias(a[d]);this.forwardFeed_();return this.getOutputs()};NeuralNetwork.prototype.getOutputs=function(){return this.outputLayer_.getNeurons().map(function(a){return a.getOutput()})};function Neuron(a,b){this.errorDelta_=this.error_=0;this.inputNeurons_=[];this.inputWeights_=[];this.lastWeightChanges_=[];this.outputValue_=0;if(b){var d=new Neuron(null,!1);d.setOutput(1);this.inputNeurons_.push(d);this.inputWeights_.push(2*Math.random()-1);this.lastWeightChanges_.push(0);this.outputValue_=-1}a&&this.setInputs(a)}Neuron.prototype.setError=function(a){this.error_=a};
Neuron.prototype.updateWeights=function(a,b){for(var d=this.errorDelta_,c=1;c<this.inputNeurons_.length;c++){var e=this.lastWeightChanges_[c],f=this.inputNeurons_[c].getOutput(),e=a*d*f+b*e;this.lastWeightChanges_[c]=e;this.inputWeights_[c]+=e}this.inputWeights_[0]+=a*d};Neuron.prototype.setErrorDelta=function(a){this.errorDelta_=a};Neuron.prototype.getError=function(){return this.error_};Neuron.prototype.getErrorDelta=function(){return this.errorDelta_};Neuron.prototype.getWeight=function(a){return this.inputWeights_[a]};
Neuron.prototype.setWeight=function(a,b){this.inputWeights_[a]=b};Neuron.prototype.setWeightChange=function(a,b){this.lastWeightChanges_[a]=b};Neuron.prototype.getOutput=function(){return this.outputValue_};Neuron.prototype.setOutput=function(a){this.outputValue_=a};Neuron.prototype.synapse=function(){for(var a=0,b=0;b<this.inputNeurons_.length;b++)var d=this.inputNeurons_[b].getOutput(),a=a+d*this.inputWeights_[b];this.outputValue_=1/(1+Math.exp(-a))};
Neuron.prototype.setBias=function(a){this.inputWeights_[0]=a};Neuron.prototype.setInputs=function(a){var b=this.inputWeights_[0];this.inputNeurons_=[this.inputNeurons_[0]];this.inputWeights_=[b];this.lastWeightChanges_=[0];for(b=0;b<a.length;b++){var d=2*Math.random()-1;this.inputNeurons_.push(a[b]);this.inputWeights_.push(d);this.lastWeightChanges_.push(0)}};