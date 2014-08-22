/**
 * A artificial neuron that connects to other neurons.
 * @param {Array.<Neuron>}  inputs  The connected neurons
 * @param {boolean} hasBias Whether or not this neuron has a 
 *                          bias.
 */
function Neuron(inputs, hasBias) {
	this.hasBias_ = hasBias;
	this.error_ = 0.0;

	this.inputNeurons_ = [];
	this.inputWeights_ = [];
	this.lastWeightChanges_ = [];

	this.outputValue_ = 0.0;

	if (hasBias) {
		var bias = new Neuron(null, false);
		bias.setOutput(1.0);

		this.inputNeurons_.push(bias);
		this.inputWeights_.push((Math.random() * 2) - 1);
		this.lastWeightChanges_.push(0.0);
		this.outputValue_ = -1.0;
	}

	if (inputs) {
		this.setInputs(inputs);
	}
}


/**
 * Sets the error of this neuron.
 * @param {float} givenError The error to set
 */
Neuron.prototype.setError = function(givenError) {
	this.error_ = givenError;
};


/**
 * Updates the weights of this neuron's inputs
 * @param  {float} learningRate The rate at which to learn
 * @param  {float} momentum     The momentum of the change
 */
Neuron.prototype.updateWeights = function(learningRate, momentum) {
	var delta = this.outputValue_ * (1 - this.outputValue_) * this.error_;
	var startNeuron = 0;
	if (this.hasBias_) {
		// Set the bias
		this.inputWeights_[0] += learningRate * delta;
		startNeuron = 1;
	}
	for (var i = startNeuron; i < this.inputNeurons_.length; i++) {
		var lastChange = this.lastWeightChanges_[i];
		var incomingOutput = this.inputNeurons_[i].getOutput();
		var change =
			(learningRate * delta * incomingOutput);// +
			// (momentum * lastChange);
		this.lastWeightChanges_[i] = change;
		this.inputWeights_[i] += change;
	}
};


/**
 * Gets the error of this neuron.
 * @return {float} The error of this neuron
 */
Neuron.prototype.getError = function() {
	return this.error_;
};


/**
 * Gets the weight of this neuron at a specific index.
 * @param  {int} index The index of the incoming neural connection
 * @return {float}   The weight at this index.
 */
Neuron.prototype.getWeight = function(index) {
	return this.inputWeights_[index];
};


/**
 * Sets the input weight at a specific index in this neuron.
 * @param  {int} index The index of the incoming neural connection
 * @param {float} value The weight to set for the neural connection
 */
Neuron.prototype.setWeight = function(index, value) {
	this.inputWeights_[index] = value;
};


/**
 * Sets the change in weight of this neuron at a specific index.
 * @param  {int} index The index of the incoming neural connection
 * @param {float} value The change in weight to set for the neural connection
 */
Neuron.prototype.setWeightChange = function(index, value) {
	this.lastWeightChanges_[index] = value;
};


/**
 * Gets the current output value of this neuron.
 * @return {float} The current output value
 */
Neuron.prototype.getOutput = function() {
	return this.outputValue_;
};


/**
 * Sets the current output value of this neuron
 * @param {float} output The new output value
 */
Neuron.prototype.setOutput = function(output) {
	this.outputValue_ = output;
};


/**
 * Synapses the current neuron (calculates the output)
 */
Neuron.prototype.synapse = function() {
	var sum = 0;
	for (var i = 0; i < this.inputNeurons_.length; i++) {
		var input = this.inputNeurons_[i].getOutput();
		var weight = this.inputWeights_[i];
		sum += input * weight;
	}
	// Sigmoid function
	this.outputValue_ = 1 / (1 + Math.exp(-sum));
};


/**
 * Sets the bias of this neuron
 * @param {float} bias The bias of the neuron
 */
Neuron.prototype.setBias = function(bias) {
	this.inputWeights_[0] = bias;
};


/**
 * Sets the inputs of this neuron
 * @param {Array.<Neuron>} inputs The new neurons to connect to.
 */
Neuron.prototype.setInputs = function(inputs) {
	var bias = this.inputNeurons_[0];
	var biasWeight = this.inputWeights_[0];
	this.inputNeurons_ = [bias];
	this.inputWeights_ = [biasWeight];
	this.lastWeightChanges_ = [0.0];

	for (var i = 0; i < inputs.length; i++) {
		var weight = (Math.random() * 2) - 1;
		this.inputNeurons_.push(inputs[i]);
		this.inputWeights_.push(weight);
		this.lastWeightChanges_.push(0.0);
	}
};