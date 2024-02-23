#include <NN.hpp>
#include <random>
#include <exception>
#include <algorithm>

#pragma warning(push)
#pragma warning(disable: 4996)
#pragma warning(disable: 4267) 
#pragma warning(disable: 4244) 

inline double     getRandomWeight();
inline void       init(NN::Matrix& mat, int rows, int cols);
inline NN::Matrix transpose(const NN::Matrix& mat);
inline NN::Matrix dot(const NN::Matrix& mat1, const NN::Matrix& mat2);
inline NN::Matrix tanh(const NN::Matrix& m);
inline NN::Matrix tanhPrime(const NN::Matrix& m);
inline NN::Matrix operator+(const NN::Matrix& l, const NN::Matrix& r) {
	size_t arows = l.size(), acols = l[0].size(), brows = r.size(), bcols = r[0].size();
	NN::Matrix output;
	if ((arows != brows) || (acols != bcols))
		throw std::runtime_error("Cannot add two matrecis with different dimensoins");
	init(output, arows, acols);
	for (int i = 0; i < arows; ++i)
		for (int j = 0; j < acols; ++j)
			output[i][j] = l[i][j] + r[i][j];
	return output;
}
inline NN::Matrix operator*(const NN::Matrix& l, const NN::Matrix& r) {
	size_t arow = l.size(), acols = l[0].size(), brow = r.size(), bcols = r[0].size();
	if ((arow != brow) || (acols != bcols))
		throw std::runtime_error("Cannot multiply two matrecis with different dimensions!");
	NN::Matrix output;
	init(output, arow, acols);
	for (int i = 0; i < arow; ++i)
		for (int j = 0; j < brow; ++j)
			output[i][j] = l[i][j] * r[i][j];
	return output;
}
inline NN::Matrix operator*(const NN::Matrix& l, const double  r) {
	size_t arow = l.size(), acols = l[0].size();
	NN::Matrix output;
	init(output, arow, acols);
	for (int i = 0; i < arow; ++i)
		for (int j = 0; j < acols; ++j)
			output[i][j] = l[i][j] * r;
	return output;
}
inline NN::Matrix operator+(const NN::Matrix& l, const double  r) {
	size_t arows = l.size(), acols = l[0].size();
	NN::Matrix output;
	init(output, arows, acols);
	for (int i = 0; i < arows; ++i)
		for (int j = 0; j < acols; ++j)
			output[i][j] = l[i][j] + r;
	return output;
}
inline NN::Matrix operator-(const NN::Matrix& l, const double  r)
{
	size_t arow = l.size(), acols = l[0].size();
	NN::Matrix output;
	init(output, arow, acols);
	for (int i = 0; i < arow; ++i)
		for (int j = 0; j < acols; ++j)
			output[i][j] = l[i][j] - r;
	return output;
}
inline NN::Matrix operator-(const NN::Matrix& l, const NN::Matrix& r)
{
	size_t arow = l.size(), acols = l[0].size(), brow = r.size(), bcols = r[0].size();
	if ((arow != brow) || (acols != bcols))
		throw std::runtime_error("Cannot subtract two matrecis with different dimensions!");
	NN::Matrix output;
	init(output, arow, acols);
	for (int i = 0; i < arow; ++i)
		for (int j = 0; j < acols; ++j)
			output[i][j] = l[i][j] - r[i][j];
	return output;
}
inline NN::Matrix operator-(const double  l, const NN::Matrix& r)
{
	size_t arow = r.size(), acols = r[0].size();
	NN::Matrix output;
	init(output, arow, acols);
	for (int i = 0; i < arow; ++i)
		for (int j = 0; j < acols; ++j)
			output[i][j] = l - r[i][j];
	return output;
}
inline NN::Matrix operator/(const NN::Matrix& l, const double  r)
{
	size_t arow = l.size(), acols = l[0].size();
	NN::Matrix output;
	init(output, arow, acols);
	for (int i = 0; i < arow; ++i)
		for (int j = 0; j < acols; ++j)
			output[i][j] = l[i][j] / r;
	return output;
}
inline NN::Matrix operator/(const NN::Matrix& l, const NN::Matrix& r)
{
	size_t arow = l.size(), acols = l[0].size(), brow = r.size(), bcols = r[0].size();
	if ((arow != brow) || (acols != bcols))
		throw std::runtime_error("Cannot divide two matrecis with different dimensions!");
	NN::Matrix output;
	init(output, arow, acols);
	for (int i = 0; i < arow; ++i)
		for (int j = 0; j < acols; ++j)
			output[i][j] = l[i][j] / r[i][j];
	return output;
}

NN::Layer::Layer(std::pair<int, int> topology)
{
	this->topology = topology;
	for (int i = 0; i < topology.first; ++i) {
		std::vector<double> w(topology.second);
		for (int j = 0; j < topology.second; ++j) {
			w[j] = getRandomWeight();
		}
		this->weights.push_back(w);
	}
	this->bias.push_back(std::vector<double>(topology.second, 0));
	for (int i = 0; i < bias[0].size(); ++i)
		bias[0][i] = getRandomWeight();
}

NN::Layer::Layer(FILE* file, std::pair<int, int> topology)
{
	this->topology = topology;
	for (int i = 0; i < topology.first; ++i) {
		this->weights.push_back(std::vector<double>(topology.second));
		for (int j = 0; j < topology.second; ++j) {
			double buff;
			if (!fread(&buff, sizeof(buff), 1, file) == 1) goto _a7a;
			this->weights[i].push_back(buff);
		}
	}
	this->bias.push_back(std::vector<double>(topology.second));
	for (int i = 0; i < topology.second; ++i) {
		double buff;
		if (fread(&buff, sizeof(buff), 1, file) == 1) goto _a7a;
		this->bias[0][i] = buff;
	}
_a7a:
	throw std::runtime_error("Cannot read file!");
}

NN::Matrix NN::Layer::forward(const NN::Matrix& input)
{
	this->input = input;
	Matrix output = dot(input, this->weights) + this->bias;
	return output;
}

NN::Matrix NN::Layer::backward(const NN::Matrix& error, const double lr)
{
	NN::Matrix inputError = transpose(this->weights) * error;
	this->weights = this->weights - (error * transpose(this->input)) * lr;
	this->bias = this->bias - error * lr;
	return inputError;
}

void NN::Layer::save(FILE* file)
{
	for (int i = 0; i < this->weights.size(); ++i) {
		for (int j = 0; j < this->weights[i].size(); ++j) {
			double buff = this->weights[i][j];
			if (fwrite(&buff, sizeof(double), 1, file) != 1) goto _a7a2;
		}
	}
	for (int i = 0; i < this->bias[0].size(); ++i) {
		double buff = this->bias[0][i];
		if (!fwrite(&buff, sizeof(double), 1, file) != 1) goto _a7a2;
	}
	return;
_a7a2:
	throw std::runtime_error("Failed to save network!");
}

NN::ActivationLayer::ActivationLayer() {} // here to make the compiler happy.

NN::Matrix NN::ActivationLayer::forward(const NN::Matrix& input)
{
	this->input = input;
	NN::Matrix output = tanh(input);
	return output;
}

NN::Matrix NN::ActivationLayer::backward(const NN::Matrix& error, const double lr)
{
	return tanhPrime(this->input) * error;
}

NN::Net::Net(std::initializer_list<int> topology)
{
	for (int num : topology) {
		if ((num <= 0))
			throw std::runtime_error("Cannot create a Layer with zero or less neurons!");
		this->topology.push_back(num);
	}
	size_t size = this->topology.size();
	for (int i = 1; i < size; ++i) {
		this->layers.push_back(new NN::Layer({ i - 1, i }));
		this->layers.push_back(new NN::ActivationLayer());
	}
}

NN::Net::Net(FILE* file)
{
	int buffer;
	while ((fread(&buffer, sizeof(int), 1, file) == 1) && buffer != 0)
		this->topology.push_back(buffer);
	size_t size = this->topology.size();
	for (int i = 1; i < size; ++i) {
		this->layers.push_back(new NN::Layer(file, { i - 1, i }));
		this->layers.push_back(new NN::ActivationLayer());
	}
}

NN::Matrix NN::Net::predict(const NN::Matrix& input)
{
	NN::Matrix prediction = input;
	for (auto layer : this->layers)
		prediction = layer->forward(prediction);
	return prediction;
}

void NN::Net::fit(const NN::Matrix& input, const NN::Matrix& data, const double lr)
{
	if (!this->loss || !this->loss_prime)
		throw std::runtime_error("Call Net::use(Loss, LossPrime) first!");
	std::vector<__BaseLayer*> reversedLayers = this->layers; reverse(reversedLayers.begin(), reversedLayers.end());
	NN::Matrix error = this->loss_prime(this->predict(input), data);
	for (auto layer : reversedLayers)
		error = layer->backward(error, lr);
}

void NN::Net::save(const char* filename)
{
	FILE* file;
	if (!(file = fopen(filename, "wb")))
		throw std::runtime_error("Couldn't open file!");
	for (int t : this->topology)
		fwrite(&t, sizeof(int), 1, file);
	fputc('\0', file);
	for (auto layer : this->layers)
		layer->save(file);
	fclose(file);
}

NN::Net NN::Net::load(const char* filename)
{
	FILE* file;
	if (!(file = fopen(filename, "rb")))
		throw std::runtime_error("Failed to open network file!");
	NN::Net net = NN::Net(file);
	fclose(file);
	return net;
}

void NN::Net::use(NN::Loss loss, NN::LossPrime loss_prime)
{
	this->loss = loss; this->loss_prime = loss_prime;
}

NN::Net::~Net()
{
	for (auto layer : this->layers)
		delete layer;
}

inline NN::Matrix dot(const NN::Matrix& mat1, const NN::Matrix& mat2)
{
	size_t a1 = mat1.size(), a2 = mat2.size(), b2 = mat2[0].size();
	if (a1 != b2)
		throw std::runtime_error("Cannot multiply two matrecis with different dimentions!");
	NN::Matrix output;
	init(output, a1, b2);
	for (int i = 0; i < a1; ++i)
		for (int j = 0; j < b2; ++j)
			for (int k = 0; k < a2; ++k)
				output[i][j] = mat1[i][k] * mat2[k][j];
	return output;
}

inline void init(NN::Matrix& mat, int rows, int cols)
{
	for (int i = 0; i < rows; ++i)
		mat.push_back(std::vector<double>(cols, 0.0));
}

inline double getRandomWeight()
{
	static std::random_device rd;
	static std::lognormal_distribution<double> distribution(-2.0, 2.0);
	return distribution(rd);
}

inline NN::Matrix transpose(const NN::Matrix& mat)
{
	NN::Matrix output;
	size_t rows = mat.size(), cols = mat.size();
	init(output, rows, cols);
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			output[i][j] = mat[j][i];
	return output;
}

inline NN::Matrix tanh(const NN::Matrix& m)
{
	NN::Matrix output;
	size_t arow = m.size(), acol = m[0].size();
	init(output, arow, acol);
	for (int i = 0; i < arow; ++i)
		for (int j = 0; j < acol; ++j)
			output[i][j] = tanh(m[i][j]);
	return output;
}

inline NN::Matrix tanhPrime(const NN::Matrix& m)
{
	NN::Matrix t = tanh(m);
	return 1 - t * t;
}

NN::Matrix NN::mse(const NN::Matrix& predicted, const NN::Matrix& actual)
{
	NN::Matrix output = actual - predicted;
	return (output * output) * 0.5;
}

NN::Matrix NN::msePrime(const NN::Matrix& predicted, const NN::Matrix& actual)
{
	return  (predicted - actual) * 2 / actual.size();
}

#pragma warning(pop)