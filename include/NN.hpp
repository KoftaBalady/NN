#pragma once

#include <vector>
#include <functional>

#define NN_FLT_EPSILON 1E-5
#define CMP(x, y)                         \
	(fabsf((x) - (y)) <= NN_FLT_EPSILON * \
							 fmaxf(1.0f, fmaxf(fabsf(x), fabsf(y))))

namespace NN
{
	typedef std::vector<std::vector<double>> Matrix;
	 inline Matrix operator+(const Matrix& l, const Matrix& r);
	 inline Matrix operator+(const Matrix& l, const double r);
	 inline Matrix operator*(const Matrix& l, const Matrix& r);
	 inline Matrix operator*(const Matrix& l, const double r);
	 inline Matrix operator-(const Matrix& l, const Matrix& r);
	 inline Matrix operator-(const Matrix& l, const double r);
	 inline Matrix operator-(const double l, const Matrix& r);
	 inline Matrix operator/(const Matrix& l, const Matrix& r);
	 inline Matrix operator/(const Matrix& l, const double r);
	typedef std::function<Matrix(const Matrix&, const Matrix&)> Loss;
	typedef std::function<Matrix(const Matrix&, const Matrix&)> LossPrime;

	class __BaseLayer
	{
	public:
		 __BaseLayer() = default;
		 virtual Matrix forward(const Matrix& input) { return {}; };
		 virtual Matrix backward(const Matrix& error, const double lr) { return {}; };
		 virtual void save(FILE* file) {};

	private:
	};

	class Layer : public __BaseLayer
	{
	public:
		 Layer(std::pair<int, int> topology);
		 Layer(FILE* file, std::pair<int, int> toplogy);
		 Matrix forward(const Matrix& input);
		 Matrix backward(const Matrix& error, const double lr);
		 void save(FILE* file);

	private:
		Matrix input;
		Matrix weights;
		Matrix bias;
		std::pair<int, int> topology;
	};

	class ActivationLayer : public __BaseLayer
	{
	public:
		 ActivationLayer();
		 Matrix forward(const Matrix& input);
		 Matrix backward(const Matrix& error, const double lr);

	private:
		Matrix input;
	};

	 class Net
	{
	public:
		 Net(std::initializer_list<int> topology);
		 // Input must be a Matrix of dimesions (1, n).
		 Matrix predict(const Matrix& input);
		 void fit(const Matrix& input, const Matrix& data, const double lr);
		 static Net load(const char* filename);
		 void save(const char* filename);
		 void use(Loss loss, LossPrime loss_prime);
		 ~Net();

	private:
		 Net(FILE* file);
		std::vector<int> topology;
		std::vector<__BaseLayer*> layers;
		Loss loss;
		LossPrime loss_prime;
	};

	// Mean squared error.
	 Matrix mse(const Matrix& predicted, const Matrix& actual);
	// The derivative for the mean squared error.
	 Matrix msePrime(const Matrix& predicted, const Matrix& actual);

} // namespace NN